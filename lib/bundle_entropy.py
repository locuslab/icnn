import numpy as np
import scipy as sp
import numpy.random as npr

def pdipm_pc(G, h, z0=None):
    # Mehrotra's PC method as described in the cvxgen paper:
    # http://stanford.edu/~boyd/papers/pdf/code_gen_impl.pdf

    nDual, nPrimal = G.shape
    z = z0.copy() if z0 is not None else np.ones(nDual)/nDual
    y = np.full(nPrimal, 0.5) #1/(1+np.exp(A.T.dot(z)))
    s = np.ones(nDual)
    t = 1.

    for i in range(20):
        grad_negH = np.log(y) - np.log(1.-y)
        hess_negH = np.diag(1./y + 1./(1.-y))
        hess_negH_inv = np.diag(1./(1./y + 1./(1.-y)))

        # A = np.bmat([[hess_negH, np.zeros((nPrimal, 1+nDual)), G.T],
        #             [np.zeros((1,nPrimal+1+nDual)), -np.ones((1,nDual))],
        #             [np.zeros((nDual,nPrimal+1)), np.diag(z/s), np.eye(nDual)],
        #             [G, -np.ones((nDual, 1)), np.eye(nDual), np.zeros((nDual, nDual))]])

        ry = grad_negH + G.T.dot(z) # y residual.
        rt = 1.-np.sum(z) # t residual.
        rc = z # Complementary slackness residual.
        rd = G.dot(y) + h - t*np.ones(nDual) + s # Dual residual.

        mu = s.dot(z)/nDual
        pri_res = np.linalg.norm(np.concatenate([ry, [rt]]))
        dual_res = np.linalg.norm(rd)
        d = z/s
        print(("primal_res = {0:.5g}, dual_res = {1:.5g}, " +
                "gap = {2:.5g}, kappa(d) = {3:.5g}").format(
                    pri_res, dual_res, mu, min(d)/max(d)))

        if pri_res < 1e-8 and dual_res < 1e-8:
            return y, z

        M = G.dot(hess_negH_inv).dot(G.T) + np.diag(s/z)
        chol_M = np.linalg.cholesky(M)
        Minv_1 = sp.linalg.cho_solve((chol_M, True), np.ones(nDual))

        def solve(ry, rt, rc, rd):
            r = rd - G.dot(hess_negH_inv).dot(ry) - (s/z)*rc
            dt = (r.dot(Minv_1) - rt) / Minv_1.sum()
            dz = sp.linalg.cho_solve((chol_M, True), r-dt)
            ds = -(s/z)*(rc+dz)
            dy = -hess_negH_inv.dot(ry+G.T.dot(dz))
            return dt, dz, ds, dy

        dt_aff, dz_aff, ds_aff, dy_aff = solve(ry, rt, rc, rd)

        alpha = min(get_step(z, dz_aff), get_step(s, ds_aff),
                    get_step(y, dy_aff), get_step(-y+1, -dy_aff), 1.0)
        sig = (np.dot(s + alpha*ds_aff, z + alpha*dz_aff)/(np.dot(s,z)))**3

        mu = np.dot(s,z)/nDual

        ry[:] = rt = rd[:] = 0
        rc = -(mu*sig*np.ones(nDual) - ds_aff*dz_aff)/s
        dt_cor, dz_cor, ds_cor, dy_cor = solve(ry, rt, rc, rd)

        dy = dy_aff + dy_cor
        dt = dt_aff + dt_cor
        ds = ds_aff + ds_cor
        dz = dz_aff + dz_cor

        alpha = max(0.0, min(1.0, 0.99*min(get_step(s,ds), get_step(z,dz),
                                           get_step(y, dy), get_step(-y+1, -dy))))

        y += alpha*dy
        t += alpha*dt
        s += alpha*ds
        z += alpha*dz

    return y, z

def pdipm_boyd(G, h, z0=None):
    # p612 of Boyd's Convex Optimization book.

    alpha = 0.05
    beta = 0.5
    mu = 10

    nDual, nPrimal = G.shape
    z = z0.copy() if z0 is not None else np.ones(nDual)/nDual
    y = np.full(nPrimal, 0.5) #1/(1+np.exp(A.T.dot(z)))

    # Choose t and s so that Gy + h - t + s = 0
    t = np.max(G.dot(y)+h)+1.0
    s = -G.dot(y)-h+t

    for i in range(20):
        grad_negH = np.log(y) - np.log(1.-y)
        hess_negH = np.diag(1./y + 1./(1.-y))
        hess_negH_inv = np.diag(1./(1./y + 1./(1.-y)))

        gap = s.dot(z)/nDual
        u = mu/gap # Modified complementary slackness.

        def res(y, t, s, z):
            grad_negH = np.log(y) - np.log(1.-y)
            ry = grad_negH + G.T.dot(z) # y residual.
            rt = 1.-np.sum(z) # t residual.
            rc = s*z + 1./u # Modified complementary slackness residual.
            rd = G.dot(y) + h - t*np.ones(nDual) + s # Dual residual.
            return ry, rt, rc, rd

        ry, rt, rc, rd = res(y, t, s, z)

        pri_res = np.linalg.norm(np.concatenate([ry, [rt]]))
        dual_res = np.linalg.norm(rd)
        d = z/s
        print(("primal_res = {0:.5g}, dual_res = {1:.5g}, " +
                "gap = {2:.5g}, kappa(d) = {3:.5g}").format(
                    pri_res, dual_res, gap, min(d)/max(d)))

        if pri_res < 1e-8 and dual_res < 1e-8:
            return y, z

        A = np.bmat([[hess_negH, np.zeros((nPrimal, 1+nDual)), G.T],
                    [np.zeros((1,nPrimal+1+nDual)), -np.ones((1,nDual))],
                    [np.zeros((nDual,nPrimal+1)), np.diag(z), np.diag(s)],
                    [G, -np.ones((nDual, 1)), np.eye(nDual), np.zeros((nDual, nDual))]])

        r = np.concatenate([ry, [rt], rc, rd])
        d = np.linalg.solve(A, -r)
        dy, dt, ds, dz = np.split(d, [nPrimal, nPrimal+1, nPrimal+1+nDual])

        step = min(1.0, 0.99*min(get_step(s, ds), get_step(z, dz),
                                 get_step(y, dy), get_step(-y+1, -dy)))

        def update(step):
            return y + step*dy, t + step*dt, s + step*ds, z + step*dz

        def f(step):
            yp, tp, sp, zp = update(step)
            return np.all(G.dot(yp)+h-tp+sp >= 0)

        def g(step):
            yp, tp, sp, zp = update(step)
            ry, rt, rc, rd = res(yp, tp, sp, zp)
            rp = np.concatenate([ry, [rt], rc, rd])
            return np.linalg.norm(rp) > (1.-alpha*step)*np.linalg.norm(r)

        while f(step):
            step *= beta

        while g(step):
            step *= beta

        y, t, s, z = update(step)

    return y, z

def get_step(v,dv):
    if np.any(dv<0):
        a = -v/dv
        return np.min(a[dv<0])
    else:
        return 1.

def negH(y):
    return np.sum(y*np.log(y) + (1.-y)*np.log(1.-y))

def solve(fg, initX, nIter=10, callback=None):
    A = []
    b = []

    x = initX
    err = []

    for t in range(nIter):
        fi, gi = fg(x)
        Ai = gi
        bi = fi - np.dot(gi, x)
        A.append(Ai)
        b.append(bi)

        if callback is not None:
            callback(t, fi, x)

        x, lam = pdipm(G, h)

        A = [y for i,y in enumerate(A) if lam[i] > 0]
        b = [y for i,y in enumerate(b) if lam[i] > 0]

    return x

def solveBatch(fg, initXs, nIter=10, callback=None, solver='pc'):
    bsize = initXs.shape[0]
    A = [[] for i in range(bsize)]
    b = [[] for i in range(bsize)]
    xs = [[] for i in range(bsize)]
    lam = [None]*bsize
    eps = 1e-8

    x = initXs

    finished = []
    nIters = [nIter]*bsize
    for t in range(nIter):
        fi, gi = fg(x)
        Ai = gi
        bi = fi - np.sum(gi * x, axis=1)
        if callback is not None:
            callback(t, fi, x)

        for u in range(bsize):
            if u in finished:
                continue

            A[u].append(Ai[u])
            b[u].append(bi[u])
            xs[u].append(np.copy(x[u]))

            if np.linalg.matrix_rank(np.array(A[u])) < len(A[u]):
                del(A[u][-1])
                del(b[u][-1])
                del(xs[u][-1])
                finished.append(u)
                nIters[u] = t-1
                continue

            if solver == 'pc':
                x[u], lam[u] = pdipm_pc(np.array(A[u]), np.array(b[u]))
            elif solver == 'boyd':
                x[u], lam[u] = pdipm_boyd(np.array(A[u]), np.array(b[u]))
            else:
                raise RuntimeError("Solver unknown: "+solver)

            A[u] = [y for i,y in enumerate(A[u]) if lam[u][i] > eps]
            b[u] = [y for i,y in enumerate(b[u]) if lam[u][i] > eps]
            xs[u] = [y for i,y in enumerate(xs[u]) if lam[u][i] > eps]
            lam[u] = lam[u][lam[u] > eps]

        if len(finished) == bsize:
            return x, A, b, lam, xs, nIters

    return x, A, b, lam, xs, nIters
