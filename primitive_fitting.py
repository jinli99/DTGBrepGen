import torch
import numpy as np
import scipy
from scipy.optimize import minimize
from torch.autograd import Function
from weakref import WeakKeyDictionary
import open3d as o3d


EPS = np.finfo(np.float32).eps
DTYPE = np.float64


def svd_grad_K(S):
    N = S.shape[0]
    s1 = S.view((1, N))
    s2 = S.view((N, 1))
    diff = s2 - s1
    plus = s2 + s1

    device = S.device
    eps = torch.ones((N, N), device=device) * 10 ** (-6)
    max_diff = torch.max(torch.abs(diff), eps)
    sign_diff = torch.sign(diff)

    K_neg = sign_diff * max_diff

    # guard the matrix inversion
    K_neg[torch.arange(N), torch.arange(N)] = 10 ** (-6)
    K_neg = 1 / K_neg
    K_pos = 1 / plus

    ones = torch.ones((N, N), device=device)
    rm_diag = ones - torch.eye(N, device=device)
    K = K_neg * K_pos * rm_diag
    return K


def compute_grad_V(U, S, V, grad_V):
    N = S.shape[0]
    K = svd_grad_K(S)
    device = S.device
    S = torch.eye(N, device=device) * S.reshape((N, 1))
    inner = K.T * (V.T @ grad_V)
    inner = (inner + inner.T) / 2.0
    return 2 * U @ S @ inner @ V.T


class CustomSVD(Function):
    """
    Costum SVD to deal with the situations when the
    singular values are equal. In this case, if dealt
    normally the gradient w.r.t to the input goes to inf.
    To deal with this situation, we replace the entries of
    a K matrix from eq: 13 in https://arxiv.org/pdf/1509.07838.pdf
    to high value.
    Note: only applicable for the tall and square matrix and doesn't
    give correct gradients for fat matrix. Maybe transpose of the
    original matrix is requiring to deal with this situation. Left for
    future work.
    """

    @staticmethod
    def forward(ctx, input):
        # Note: input is matrix of size m x n with m >= n.
        # Note: if above assumption is voilated, the gradients
        # will be wrong.
        U, S, V = torch.svd(input, some=True)

        ctx.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(ctx, grad_U, grad_S, grad_V):
        U, S, V = ctx.saved_tensors
        grad_input = compute_grad_V(U, S, V, grad_V)
        return grad_input


customsvd = CustomSVD.apply


def best_lambda(A):
    """
    Takes an under determined system and small lambda value,
    and comes up with lambda that makes the matrix (A + lambda*I) invertible.
    Assuming A to be square matrix.
    """
    lamb = 1e-6
    cols = A.shape[0]

    for i in range(7):
        A_dash = A + lamb * torch.eye(cols, device=A.get_device())
        if cols == torch.linalg.matrix_rank(A_dash):
            # we achieved the required rank
            break
        else:
            # factor by which to increase the lambda. Choosing 10 for performance.
            lamb *= 10
    return lamb


def get_rotation_matrix(theta):
    R = np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return R


def rotation_matrix_a_to_b(A, B):
    """
    Finds rotation matrix from vector A in 3d to vector B
    in 3d.
    B = R @ A
    """
    cos = np.dot(A, B)
    sin = np.linalg.norm(np.cross(B, A))
    u = A
    v = B - np.dot(A, B) * A
    v = v / (np.linalg.norm(v) + EPS)
    w = np.cross(B, A)
    w = w / (np.linalg.norm(w) + EPS)
    F = np.stack([u, v, w], 1)
    G = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    # B = R @ A
    try:
        R = F @ G @ np.linalg.inv(F)
    except:
        R = np.eye(3, dtype=np.float32)
    return R


def regular_parameterization(grid_u, grid_v):
    nx, ny = (grid_u, grid_v)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    xv = np.expand_dims(xv.transpose().flatten(), 1)
    yv = np.expand_dims(yv.transpose().flatten(), 1)
    parameters = np.concatenate([xv, yv], 1)
    return parameters


def fitcylinder(data, guess_angles=None):
    """Fit a list of data points to a cylinder surface. The algorithm implemented
    here is from David Eberly's paper "Fitting 3D Data with a Cylinder" from
    https://www.geometrictools.com/Documentation/CylinderFitting.pdf
    Arguments:
        data - A list of 3D data points to be fitted.
        guess_angles[0] - Guess of the theta angle of the axis direction
        guess_angles[1] - Guess of the phi angle of the axis direction

    Return:
        Direction of the cylinder axis
        A point on the cylinder axis
        Radius of the cylinder
        Fitting error (G function)
    """

    def direction(theta, phi):
        """Return the direction vector of a cylinder defined
        by the spherical coordinates theta and phi.
        """
        return np.array(
            [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]
        )

    def projection_matrix(w):
        """Return the projection matrix  of a direction w."""
        return np.identity(3) - np.dot(np.reshape(w, (3, 1)), np.reshape(w, (1, 3)))

    def skew_matrix(w):
        """Return the skew matrix of a direction w."""
        return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

    def calc_A(Ys):
        """Return the matrix A from a list of Y vectors."""
        return sum(np.dot(np.reshape(Y, (3, 1)), np.reshape(Y, (1, 3))) for Y in Ys)

    def calc_A_hat(A, S):
        """Return the A_hat matrix of A given the skew matrix S"""
        return np.dot(S, np.dot(A, np.transpose(S)))

    def preprocess_data(Xs_raw):
        """Translate the center of mass (COM) of the data to the origin.
        Return the prossed data and the shift of the COM"""
        n = len(Xs_raw)
        Xs_raw_mean = sum(X for X in Xs_raw) / n

        return [X - Xs_raw_mean for X in Xs_raw], Xs_raw_mean

    def G(w, Xs):
        """Calculate the G function given a cylinder direction w and a
        list of data points Xs to be fitted."""
        n = len(Xs)
        P = projection_matrix(w)
        Ys = [np.dot(P, X) for X in Xs]
        A = calc_A(Ys)
        A_hat = calc_A_hat(A, skew_matrix(w))

        u = sum(np.dot(Y, Y) for Y in Ys) / n
        v = np.dot(A_hat, sum(np.dot(Y, Y) * Y for Y in Ys)) / np.trace(
            np.dot(A_hat, A)
        )

        return sum((np.dot(Y, Y) - u - 2 * np.dot(Y, v)) ** 2 for Y in Ys)

    def C(w, Xs):
        """Calculate the cylinder center given the cylinder direction and
        a list of data points.
        """
        n = len(Xs)
        P = projection_matrix(w)
        Ys = [np.dot(P, X) for X in Xs]
        A = calc_A(Ys)
        A_hat = calc_A_hat(A, skew_matrix(w))

        return np.dot(A_hat, sum(np.dot(Y, Y) * Y for Y in Ys)) / np.trace(
            np.dot(A_hat, A)
        )

    def r(w, Xs):
        """Calculate the radius given the cylinder direction and a list
        of data points.
        """
        n = len(Xs)
        P = projection_matrix(w)
        c = C(w, Xs)

        return np.sqrt(sum(np.dot(c - X, np.dot(P, c - X)) for X in Xs) / n)

    Xs, t = preprocess_data(data)

    # Set the start points

    start_points = [(0, 0), (np.pi / 2, 0), (np.pi / 2, np.pi / 2)]
    if guess_angles:
        start_points = guess_angles

    # Fit the cylinder from different start points

    best_fit = None
    best_score = float("inf")

    for sp in start_points:
        fitted = minimize(
            lambda x: G(direction(x[0], x[1]), Xs), sp, method="Powell", tol=1e-6
        )

        if fitted.fun < best_score:
            best_score = fitted.fun
            best_fit = fitted

    w = direction(best_fit.x[0], best_fit.x[1])

    return w, C(w, Xs) + t, r(w, Xs), best_fit.fun


class Position:
    def __init__(self, dim: int):
        self.dim = dim
        self._instance_data: WeakKeyDictionary[str, np.ndarray] = WeakKeyDictionary()

    def __get__(self, instance, owner):
        try:
            view = self._instance_data[instance].view()
        except KeyError as e:
            raise AttributeError() from e
        view.flags.writeable = False
        return view

    def __set__(self, instance, value):
        value = np.array(value, dtype=DTYPE, copy=True)  # TODO copy?
        if value.shape != (self.dim,):
            raise ValueError("Could not construct a 3D point")
        self._instance_data[instance] = value


class Direction:
    def __init__(self, dim: int):
        self.dim = dim
        self._instance_data: WeakKeyDictionary[str, np.ndarray] = WeakKeyDictionary()

    def __get__(self, instance, owner):
        try:
            view = self._instance_data[instance].view()
        except KeyError as e:
            raise AttributeError() from e
        view.flags.writeable = False
        return view

    def __set__(self, instance, value):
        value = np.array(value, dtype=DTYPE, copy=True)
        value /= np.linalg.norm(value)
        if value.shape != (self.dim,):
            raise ValueError("Could not construct a 3D point")
        self._instance_data[instance] = value


def distance_point_point(p1, p2):
    return scipy.spatial.minkowski_distance(p1, p2)


def distance_line_point(line_point, line_direction, point):
    assert np.allclose(
        np.linalg.norm(line_direction), 1.0, rtol=1e-12, atol=1e-12, equal_nan=False
    )
    delta_p = point - line_point
    return distance_point_point(
        delta_p,
        np.matmul(
            np.expand_dims(np.dot(delta_p, line_direction), axis=-1),
            np.atleast_2d(line_direction),
        ),
    )


def _check_input(points, weights) -> None:
    """Check the input data of the fit functionality"""
    points = np.asarray(points)
    if weights is not None:
        weights = np.asarray(weights)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Input data has the wrong shape, expects points to be of shape ('n', 3), got {points.shape}"
        )
    if weights is not None and (weights.ndim != 1 or len(weights) != len(points)):
        raise ValueError(
            "Shape of weights does not match points, weights should be a 1 dimensional array of len(points)"
        )


class Cone:
    vertex = Position(3)
    axis = Direction(3)

    def __init__(self, theta, axis, vertex):
        self.vertex = vertex
        self.axis = axis
        self.theta = theta

    def __repr__(self):
        return f"Cone (vertex={self.vertex}, axis={self.axis}, theta={self.theta}"

    def distance_to_point(self, point):
        a = distance_line_point(self.vertex, self.axis, point)
        k = a * np.tan(self.theta)
        b = k + np.abs(np.dot((point - self.vertex), self.axis))
        l = b * np.sin(self.theta)
        d = a / np.cos(self.theta) - l  # np.abs

        return np.abs(d)


def fitcone(points, weights=None, initial_guess: Cone = None):
    """Fits a cone through a set of points"""
    _check_input(points, weights)
    initial_guesses = [
        Cone(0.0, np.array([1.0, 0, 0]), np.zeros(3)),
        Cone(0.0, np.array([0, 1.0, 0]), np.zeros(3)),
        Cone(0.0, np.array([0, 0, 1.0]), np.zeros(3)),
    ]
    if initial_guesses is None:
        raise NotImplementedError

    def cone_fit_residuals(cone_params, points, weights):
        cone = Cone(cone_params[0], cone_params[1:4], cone_params[4:7])

        distances = cone.distance_to_point(points)

        if weights is None:
            return distances

        return distances * np.sqrt(weights)

    best_fit = None
    best_score = float("inf")
    failure = False

    for initial_guess in initial_guesses:
        x0 = np.concatenate(
            [np.array([initial_guess.theta]), initial_guess.axis, initial_guess.vertex]
        )
        results = scipy.optimize.least_squares(
            cone_fit_residuals, x0=x0, args=(points, weights), ftol=1e-10
        )

        if not results.success:
            # return RuntimeError(results.message)
            continue

        if results.fun.sum() < best_score:
            best_score = results.fun.sum()
            best_fit = results

    try:
        apex = best_fit.x[4:7]
        axis = best_fit.x[1:4]
        theta = best_fit.x[0]
        err = best_fit.fun.mean()
    except:
        return None, None, None, None, True

    for iter in range(5):
        # check if the cone is valid
        c = apex.reshape((3))
        a = axis.reshape((3))
        norm_a = np.linalg.norm(a)
        a = a / norm_a
        proj = (points - c.reshape(1, 3)) @ a
        if np.max(proj) * np.min(proj) > 0:
            break
        else:
            r_max = distance_line_point(c, a, points[np.argmax(proj)])
            r_min = distance_line_point(c, a, points[np.argmin(proj)])
            h = np.max(proj) - np.min(proj)
            tan_theta2 = (r_max - r_min) / h
            r0 = distance_line_point(c, a, points[np.argmin(proj**2)])
            if tan_theta2 < 0:
                tan_theta2 = (r_min - r_max) / h
                vertex2 = c + a * (r0 / tan_theta2 + iter * 0.5)
            else:
                vertex2 = c - a * (r0 / tan_theta2 + iter * 0.5)

            initial_guess_2 = Cone(np.arctan(tan_theta2), a, vertex2)
            x0 = np.concatenate(
                [
                    np.array([initial_guess_2.theta]),
                    initial_guess_2.axis,
                    initial_guess_2.vertex,
                ]
            )
            results = scipy.optimize.least_squares(
                cone_fit_residuals, x0=x0, args=(points, weights), ftol=1e-10
            )

            if not results.success and iter != 4:
                continue
            if not results.success and iter == 4:
                failure = True
                # print('failure!')

            apex = results.x[4:7]
            axis = results.x[1:4]
            theta = results.x[0]
            err = results.fun.mean()

    return apex, axis, theta, err, failure


def guard_sqrt(x, minimum=1e-5):
    x = torch.clamp(x, min=minimum)
    return torch.sqrt(x)


class LeastSquares:
    def __init__(self):
        pass

    def lstsq(self, A, Y, lamb=0.0):
        """
        Differentiable least square
        :param A: m x n
        :param Y: n x 1
        """
        cols = A.shape[1]
        if np.isinf(A.data.cpu().numpy()).any():
            raise RuntimeError("Infinity in least squares")

        # Assuming A to be full column rank
        if cols == torch.linalg.matrix_rank(A):
            # Full column rank
            q, r = torch.linalg.qr(A)
            x = torch.inverse(r) @ q.transpose(1, 0) @ Y
        else:
            # rank(A) < n, do regularized least square.
            AtA = A.transpose(1, 0) @ A

            # get the smallest lambda that suits our purpose, so that error in
            # results minimized.
            with torch.no_grad():
                lamb = best_lambda(AtA)
            A_dash = AtA + lamb * torch.eye(cols, device=A.get_device())
            Y_dash = A.transpose(1, 0) @ Y

            # if it still doesn't work, just set the lamb to be very high value.
            x = self.lstsq(A_dash, Y_dash, 1)
        return x


class Fit:
    def __init__(self):
        """
        Defines fitting and sampling modules for geometric primitives.
        """
        LS = LeastSquares()
        self.lstsq = LS.lstsq
        self.parameters = {}

    @staticmethod
    def sample_torus(r_major, r_minor, center, axis):
        d_theta = 60
        theta = np.arange(d_theta - 1) * 3.14 * 2 / d_theta

        theta = np.concatenate([theta, np.zeros(1)])
        circle = np.stack([np.cos(theta), np.sin(theta)], 1) * r_minor

        circle = np.concatenate([np.zeros((circle.shape[0], 1)), circle], 1)
        circle[:, 1] += r_major

        d_theta = 100
        theta = np.arange(d_theta - 1) * 3.14 * 2 / d_theta
        theta = np.concatenate([theta, np.zeros(1)])

        torus = []
        for i in range(d_theta):
            R = get_rotation_matrix(theta[i])
            torus.append((R @ circle.T).T)

        torus = np.concatenate(torus, 0)
        R = rotation_matrix_a_to_b(np.array([0, 0, 1.0]), axis)
        torus = (R @ torus.T).T
        torus = torus + center
        return torus

    @staticmethod
    def sample_plane(d, n, mean):
        regular_parameters = regular_parameterization(120, 120)
        n = n.reshape(3)
        r1 = np.random.random()
        r2 = np.random.random()
        a = (d - r1 * n[1] - r2 * n[2]) / (n[0] + EPS)
        x = np.array([a, r1, r2]) - d * n

        x = x / np.linalg.norm(x)
        n = n.reshape((1, 3))

        # let us find the perpendicular vector to a lying on the plane
        y = np.cross(x, n)
        y = y / np.linalg.norm(y)

        param = 1 - 2 * np.array(regular_parameters)
        param = param * 0.75

        gridded_points = param[:, 0:1] * x + param[:, 1:2] * y
        gridded_points = gridded_points + mean
        return gridded_points

    @staticmethod
    def sample_cone_trim(c, a, theta, points):
        """
        Trims the cone's height based points. Basically we project
        the points on the axis and retain only the points that are in
        the range.
        """
        if c is None:
            return None, None
        c = c.reshape((3))
        a = a.reshape((3))
        norm_a = np.linalg.norm(a)
        a = a / norm_a
        proj = (points - c.reshape(1, 3)) @ a
        proj_max = np.max(proj) + 0.2 * np.abs(np.max(proj))
        proj_min = np.min(proj) - 0.2 * np.abs(np.min(proj))

        # find one point on the cone
        k = np.dot(c, a)
        x = (k - a[1] - a[2]) / (a[0] + EPS)
        y = 1
        z = 1
        d = np.array([x, y, z])
        p = a * (np.linalg.norm(d)) / (np.sin(theta) + EPS) * np.cos(theta) + d

        # This is a point on the surface
        p = p.reshape((3, 1))

        # Now rotate the vector p around axis a by variable degree
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        points = []
        normals = []
        c = c.reshape((3, 1))
        a = a.reshape((3, 1))
        rel_unit_vector = p - c
        rel_unit_vector = (p - c) / np.linalg.norm(p - c)
        rel_unit_vector_min = rel_unit_vector * (proj_min) / (np.cos(theta) + EPS)
        rel_unit_vector_max = rel_unit_vector * (proj_max) / (np.cos(theta) + EPS)

        for j in range(100):
            # p_ = (p - c) * (0.01) * j
            p_ = (
                rel_unit_vector_min
                + (rel_unit_vector_max - rel_unit_vector_min) * 0.01 * j
            )

            d_points = []
            d_normals = []
            for d in range(50):
                degrees = 2 * np.pi * 0.01 * d * 2
                R = np.eye(3) + np.sin(degrees) * K + (1 - np.cos(degrees)) * K @ K
                rotate_point = R @ p_
                d_points.append(rotate_point + c)
                d_normals.append(
                    rotate_point
                    - np.linalg.norm(rotate_point) / np.cos(theta) * a / norm_a
                )

            # repeat the points to close the circle
            d_points.append(d_points[0])
            d_normals.append(d_normals[0])

            points += d_points
            normals += d_normals

        points = np.stack(points, 0)[:, :, 0]
        normals = np.stack(normals, 0)[:, :, 0]
        normals = normals / (np.expand_dims(np.linalg.norm(normals, axis=1), 1) + EPS)

        # projecting points to the axis to trim the cone along the height.
        proj = (points - c.reshape((1, 3))) @ a
        proj = proj[:, 0]
        indices = np.logical_and(proj < proj_max, proj > proj_min)
        # project points on the axis, remove points that are beyond the limits.
        return points[indices], normals[indices]

    @staticmethod
    def sample_sphere(radius, center, N=1000):
        center = center.reshape((1, 3))
        d_theta = 100
        theta = np.arange(d_theta - 1) * 3.14 * 2 / d_theta
        theta = np.concatenate([theta, np.zeros(1)])
        circle = np.stack([np.cos(theta), np.sin(theta)], 1)
        lam = np.linspace(
            -radius + 1e-7, radius - 1e-7, 100
        )  # np.linspace(-1 + 1e-7, 1 - 1e-7, 100)
        radii = np.sqrt(radius**2 - lam**2)  # radius * np.sqrt(1 - lam ** 2)
        circle = np.concatenate([circle] * lam.shape[0], 0)
        spread_radii = np.repeat(radii, d_theta, 0)
        new_circle = circle * spread_radii.reshape((-1, 1))
        height = np.repeat(lam, d_theta, 0)
        points = np.concatenate([new_circle, height.reshape((-1, 1))], 1)
        points = points - np.mean(points, 0)
        normals = points / np.linalg.norm(points, axis=1, keepdims=True)
        points = points + center
        return points, normals

    @staticmethod
    def sample_cylinder_trim(radius, center, axis, points, N=1000):
        """
        :param center: center of size 1 x 3
        :param radius: radius of the cylinder
        :param axis: axis of the cylinder, size 3 x 1
        """
        center = center.reshape((1, 3))
        axis = axis.reshape((3, 1))

        d_theta = 60
        d_height = 100

        R = rotation_matrix_a_to_b(np.array([0, 0, 1]), axis[:, 0])

        # project points on to the axis
        points = points - center

        projection = points @ axis
        arg_min_proj = np.argmin(projection)
        arg_max_proj = np.argmax(projection)

        min_proj = np.squeeze(projection[arg_min_proj]) - 0.1
        max_proj = np.squeeze(projection[arg_max_proj]) + 0.1

        theta = np.arange(d_theta - 1) * 3.14 * 2 / d_theta

        theta = np.concatenate([theta, np.zeros(1)])
        circle = np.stack([np.cos(theta), np.sin(theta)], 1)
        circle = np.concatenate([circle] * 2 * d_height, 0) * radius

        normals = np.concatenate([circle, np.zeros((circle.shape[0], 1))], 1)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        height = np.expand_dims(np.linspace(min_proj, max_proj, 2 * d_height), 1)
        height = np.repeat(height, d_theta, axis=0)
        points = np.concatenate([circle, height], 1)
        points = R @ points.T
        points = points.T + center
        normals = (R @ normals.T).T

        return points, normals

    @staticmethod
    def fit_plane_torch(points, normals, weights, ids=0, show_warning=False):
        """
        Fits plane
        :param points: points with size N x 3
        :param weights: weights with size N x 1
        """
        weights_sum = torch.sum(weights) + EPS

        X = points - torch.sum(weights * points, 0).reshape((1, 3)) / weights_sum

        weighted_X = weights * X
        np_weighted_X = weighted_X.data.cpu().numpy()
        if np.linalg.cond(np_weighted_X) > 1e5:
            if show_warning:
                print("condition number is large in plane!", np.sum(np_weighted_X))
                print(torch.sum(points), torch.sum(weights))

        U, s, V = customsvd(weighted_X)
        a = V[:, -1]
        a = torch.reshape(a, (1, 3))
        d = torch.sum(weights * (a @ points.permute(1, 0)).permute(1, 0)) / weights_sum
        return a, d

    def fit_sphere_torch(self, points, normals, weights, ids=0, show_warning=False):
        N = weights.shape[0]
        sum_weights = torch.sum(weights) + EPS
        A = 2 * (-points + torch.sum(points * weights, 0) / sum_weights)

        dot_points = weights * torch.sum(points * points, 1, keepdim=True)

        normalization = torch.sum(dot_points) / sum_weights

        Y = dot_points - normalization
        Y = Y.reshape((N, 1))
        A = weights * A
        Y = weights * Y

        if np.linalg.cond(A.data.cpu().numpy()) > 1e8:
            if show_warning:
                print("condition number is large in sphere!")

        center = -self.lstsq(A, Y, 0.01).reshape((1, 3))
        radius_square = (
            torch.sum(weights[:, 0] * torch.sum((points - center) ** 2, 1))
            / sum_weights
        )
        radius_square = torch.clamp(radius_square, min=1e-3)
        radius = guard_sqrt(radius_square)
        return center, radius

    @staticmethod
    def fit_cylinder(points, normals, weights, ids=0, show_warning=False):
        w_fit, C_fit, r_fit, fit_err = fitcylinder(points.detach().cpu().numpy())
        return w_fit, C_fit, r_fit

    @staticmethod
    def fit_cone(points, normals, weights, ids=0, show_warning=False):
        c, a, theta, err, failure = fitcone(points.detach().cpu().numpy())
        return c, a, theta, err, failure


def tessalate_points_fast(vertices, size_u, size_v, mask=None):
    """
    Given a grid points, this returns a tesselation of the grid using triangle.
    Furthermore, if the mask is given those grids are avoided.
    """

    def index_to_id(i, j, size_v):
        return i * size_v + j

    triangles = []

    for i in range(0, size_u - 1):
        for j in range(0, size_v - 1):
            if mask is not None:
                if mask[i, j] == 0:
                    continue
            tri = [
                index_to_id(i, j, size_v),
                index_to_id(i + 1, j, size_v),
                index_to_id(i + 1, j + 1, size_v),
            ]
            triangles.append(tri)
            tri = [
                index_to_id(i, j, size_v),
                index_to_id(i + 1, j + 1, size_v),
                index_to_id(i, j + 1, size_v),
            ]
            triangles.append(tri)
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    new_mesh.vertices = o3d.utility.Vector3dVector(np.stack(vertices, 0))
    new_mesh.remove_unreferenced_vertices()
    new_mesh.compute_vertex_normals()
    return new_mesh


def up_sample_points_torch_memory_efficient(points, times=1):
    """
    Upsamples points based on nearest neighbors.
    Takes two neareast neighbors and finds the centroid
    and that becomes the new point.
    :param points: N x 3
    """
    for _ in range(times):
        indices = []
        N = min(points.shape[0], 100)
        for i in range(points.shape[0] // N):
            diff_ = torch.sum(
                (
                    torch.unsqueeze(points[i * N : (i + 1) * N], 1)
                    - torch.unsqueeze(points, 0)
                )
                ** 2,
                2,
            )
            _, diff_indices = torch.topk(diff_, 5, 1, largest=False)
            indices.append(diff_indices)
        indices = torch.cat(indices, 0)
        neighbors = points[indices[:, 0:]]
        centers = torch.mean(neighbors, 1)
        points = torch.cat([points, centers])
    return points


def create_grid(inputs, grid_points, size_u, size_v, thres=0.02, device="cuda"):
    grid_points = torch.from_numpy(grid_points.astype(np.float32)).to(device)
    inputs = torch.from_numpy(inputs.astype(np.float32)).to(device)
    try:
        grid_points = grid_points.reshape((size_u + 2, size_v + 2, 3))
    except:
        grid_points = grid_points.reshape((size_u, size_v, 3))

    grid_points.permute(2, 0, 1)
    grid_points = torch.unsqueeze(grid_points, 0)

    filters = np.array(
        [[[0.25, 0.25], [0.25, 0.25]], [[0, 0], [0, 0]], [[0.0, 0.0], [0.0, 0.0]]]
    ).astype(np.float32)
    filters = np.stack([filters, np.roll(filters, 1, 0), np.roll(filters, 2, 0)])
    filters = torch.from_numpy(filters).to(device)
    grid_mean_points = torch.nn.functional.conv2d(grid_points.permute(0, 3, 1, 2), filters, padding=0)
    grid_mean_points = grid_mean_points.permute(0, 2, 3, 1)
    try:
        grid_mean_points = grid_mean_points.reshape(((size_u + 1) * (size_v + 1), 3))
    except:
        grid_mean_points = grid_mean_points.reshape(((size_u - 1) * (size_v - 1), 3))

    diff = []
    for i in range(grid_mean_points.shape[0]):
        diff.append(
            torch.sum(
                (
                    torch.unsqueeze(grid_mean_points[i : i + 1], 1)
                    - torch.unsqueeze(inputs, 0)
                )
                ** 2,
                2,
            )
        )
    diff = torch.cat(diff, 0)
    diff = torch.sqrt(diff)
    indices = torch.min(diff, 1)[0] < thres
    try:
        mask_grid = indices.reshape(((size_u + 1), (size_v + 1)))
    except:
        mask_grid = indices.reshape(((size_u - 1), (size_v - 1)))
    return mask_grid, diff, filter, grid_mean_points


def bit_mapping_points_torch(inputs, output_points, thres, size_u, size_v, mesh=None, device="cuda"):
    mask, diff, filters, grid_mean_points = create_grid(
        inputs, output_points, size_u, size_v, thres=thres, device=device
    )
    mesh = tessalate_points_fast(output_points, size_u, size_v, mask=mask)
    return mesh


def visualize_basic_mesh(shape_type, in_points, pred, epsilon=0.1, device="cuda"):
    if shape_type == "plane":
        # Fit plane
        part_points = (
            up_sample_points_torch_memory_efficient(in_points, 3).data.cpu().numpy()
        )
        if epsilon:
            e = epsilon
        else:
            e = 0.02
        pred_mesh = bit_mapping_points_torch(
            part_points, np.array(pred["plane_new_points"]), e, 120, 120, device=device
        )

    elif shape_type == "sphere":
        part_points = (
            up_sample_points_torch_memory_efficient(in_points, 2).data.cpu().numpy()
        )
        if epsilon:
            e = epsilon
        else:
            e = 0.03
        pred_mesh = bit_mapping_points_torch(
            part_points, np.array(pred["sphere_new_points"]), e, 100, 100, device=device
        )

    elif shape_type == "cylinder":
        part_points = (
            up_sample_points_torch_memory_efficient(in_points, 3).data.cpu().numpy()
        )

        if epsilon:
            e = epsilon
        else:
            e = 0.03
        pred_mesh = bit_mapping_points_torch(
            part_points,
            np.array(pred["cylinder_new_points"]),
            e,
            200,
            60,
            device=device,
        )

    elif shape_type == "cone":
        part_points = (
            up_sample_points_torch_memory_efficient(in_points, 3).data.cpu().numpy()
        )
        if epsilon:
            e = epsilon
        else:
            e = 0.03
        try:
            N = np.array(pred["cone_new_points"]).shape[0] // 51
            pred_mesh = bit_mapping_points_torch(
                part_points, np.array(pred["cone_new_points"]), e, N, 51, device=device
            )
        except:
            pred_mesh = None

    else:
        raise ("unseen basic shape")

    return pred_mesh


def process_one_surface(points, device, weights=None):

    points = torch.from_numpy(points).to(device)
    weights = torch.from_numpy(weights).to(device) if weights is not None else torch.ones_like(points)[:, :1]

    # ========================= fitting basic primitives =======================
    recon_basic_shapes = fit_basic_primitives(points, weights)

    # ==========================shape selection====================

    # if "cone_failure" not in recon_basic_shapes.keys():
    #     cone_err = np.inf
    # elif recon_basic_shapes["cone_failure"]:
    #     cone_err = np.inf
    # else:
    #     cone_err = recon_basic_shapes["cone_err"]

    plane_err = recon_basic_shapes["plane_err"]
    sphere_err = recon_basic_shapes["sphere_err"]
    cylinder_err = recon_basic_shapes["cylinder_err"]
    cone_err = 100

    # if (
    #     visualize_basic_mesh("cone", points, recon_basic_shapes, device=device)
    #     is None
    # ):
    #     cone_err = np.inf
    # elif (
    #     len(
    #         visualize_basic_mesh(
    #             "cone", points, recon_basic_shapes, device=device
    #         ).vertices
    #     )
    #     == 0
    # ):
    #     cone_err = np.inf
    # if "cone_params" not in recon_basic_shapes.keys():
    #     cone_err = np.inf
    # else:
    #     if recon_basic_shapes["cone_params"][2] >= 1.53:
    #         cone_err = np.inf
    if (
        len(
            visualize_basic_mesh(
                "cylinder", points, recon_basic_shapes, device=device
            ).vertices
        )
        == 0
    ):
        cylinder_err = np.inf
    if (
        len(
            visualize_basic_mesh(
                "sphere", points, recon_basic_shapes, device=device
            ).vertices
        )
        == 0
    ):
        sphere_err = np.inf

    sorted_shape_indices = np.argsort([plane_err, sphere_err, cylinder_err, cone_err])
    pred_shape = sorted_shape_indices[0]

    out = {}
    if pred_shape == 0:      # plane
        out['type'] = 'plane'
        out['params'] = recon_basic_shapes['plane_params']
        out['err'] = recon_basic_shapes['plane_err']
    elif pred_shape == 1:    # sphere
        out['type'] = 'sphere'
        out['params'] = recon_basic_shapes['sphere_params']
        out['err'] = recon_basic_shapes['sphere_err']
    elif pred_shape == 2:    # cylinder
        out['type'] = 'cylinder'
        out['params'] = recon_basic_shapes['cylinder_params']
        out['err'] = recon_basic_shapes['cylinder_err']
    elif pred_shape == 3:    # cone
        out['type'] = 'cone'
        out['params'] = recon_basic_shapes['cone_params']
        out['err'] = recon_basic_shapes['cone_err']

    return out


def project_to_plane(points, a, d):
    a = a.reshape((3, 1))
    a = a / torch.norm(a, 2)
    # Project on the same plane but passing through origin
    projections = points - ((points @ a).permute(1, 0) * a).permute(1, 0)

    # shift the points on the plane back to the original d distance
    # from origin
    projections = projections + a.transpose(1, 0) * d
    return projections


def fit_basic_primitives(pts, weights):
    """
    output: a dict of reconstructed points of each fitting shape, residual error

    """
    if pts.shape[0] < 20:
        raise ValueError("the number of points in the patch is too small")

    fitting = Fit()
    recon_basic_shapes = {}

    # ==================fit a plane=========================
    axis, distance = fitting.fit_plane_torch(
        points=pts,
        normals=None,
        weights=weights,
        ids=None,
    )
    # Project points on the surface
    new_points = project_to_plane(pts, axis, distance.item())
    plane_err = torch.linalg.norm(new_points - pts, dim=-1).mean()

    new_points = fitting.sample_plane(
        distance.item(),
        axis.data.cpu().numpy(),
        mean=torch.mean(new_points, 0).data.cpu().numpy(),
    )
    recon_basic_shapes["plane_params"] = (
        axis.reshape(-1).data.cpu().numpy().tolist(),
        distance.data.cpu().numpy().tolist(),
    )
    recon_basic_shapes["plane_new_points"] = new_points.tolist()
    recon_basic_shapes["plane_err"] = plane_err.data.cpu().numpy().tolist()

    # ======================fit a sphere======================
    center, radius = fitting.fit_sphere_torch(
        pts,
        normals=None,
        weights=weights,
        ids=None,
    )
    sphere_err = (torch.linalg.norm(pts - center, dim=-1) - radius).abs().mean()

    # Project points on the surface
    new_points, new_normals = fitting.sample_sphere(
        radius.item(), center.data.cpu().numpy(), N=10000
    )
    center = center.data.cpu().numpy()

    recon_basic_shapes["sphere_params"] = (center.reshape(-1).tolist(), radius.tolist())
    recon_basic_shapes["sphere_new_points"] = new_points.tolist()
    recon_basic_shapes["sphere_err"] = sphere_err.data.cpu().numpy().tolist()

    #======================fit a cylinder====================
    a, center, radius = fitting.fit_cylinder(
        pts,
        normals=torch.zeros_like(pts),
        weights=weights,
        ids=None,
    )

    new_points, new_normals = fitting.sample_cylinder_trim(
        radius.item(),
        center,
        a,
        pts.data.cpu().numpy(),
        N=10000,
    )
    cylinder_err = np.abs(
        (distance_line_point(center, a, pts.detach().cpu().numpy()) - radius)
    ).mean()

    recon_basic_shapes["cylinder_params"] = (
        a.tolist(),
        center.tolist(),
        radius.tolist(),
    )
    recon_basic_shapes["cylinder_new_points"] = new_points.tolist()
    recon_basic_shapes["cylinder_err"] = cylinder_err.tolist()

    # # ==========================fit a cone======================
    # apex, axis, theta, cone_err, failure = fitting.fit_cone(
    #     pts,
    #     normals=torch.zeros_like(pts),
    #     weights=weights,
    #     ids=None,
    # )
    # new_points, new_normals = fitting.sample_cone_trim(
    #     apex, axis, theta, pts.data.cpu().numpy()
    # )
    # if new_normals is not None:
    #     recon_basic_shapes["cone_params"] = (
    #         apex.tolist(),
    #         axis.tolist(),
    #         theta.tolist(),
    #     )
    #     recon_basic_shapes["cone_new_points"] = new_points.tolist()
    #     recon_basic_shapes["cone_failure"] = failure
    #     recon_basic_shapes["cone_err"] = cone_err.tolist()

    return recon_basic_shapes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = np.random.uniform(low=0.0, high=1.0, size=(1000, 3))
    points[:, -1] = 0
    out = process_one_surface(points, device)

    print(out)


if __name__ == "__main__":
    main()
