import numpy as np  # type: ignore


def balance(m, n, c, a, b):
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a == sum_b:
        print("Problem balanced!")
    else:
        if sum_a > sum_b:
            dummy = sum_a - sum_b
            b = np.append(b, dummy)
            c = np.append(c, np.zeros((m, 1)), axis=1)
            n += 1
        else:
            dummy = sum_b - sum_a
            a = np.append(a, dummy)
            c = np.append(c, np.zeros((1, n)), axis=0)
            m += 1
    return m, n, c, a, b


def least_cost_cell(m, n, c, a, b):
    temp_c = c.copy()
    temp_a = a.copy()
    temp_b = b.copy()
    X = np.zeros((m, n))
    i_index = np.array(range(m))
    j_index = np.array(range(n))
    while True:
        index = np.argmin(temp_c)
        i = index // temp_c.shape[1]
        j = index % temp_c.shape[1]
        if temp_a[i] >= temp_b[j]:
            X[i_index[i]][j_index[j]] = temp_b[j]
            temp_a[i] -= temp_b[j]
            temp_c = np.delete(temp_c, j, axis=1)
            temp_b = np.delete(temp_b, j)
            j_index = np.delete(j_index, j)
        else:
            X[i_index[i]][j_index[j]] = temp_a[i]
            temp_b[j] -= temp_a[i]
            temp_c = np.delete(temp_c, i, axis=0)
            temp_a = np.delete(temp_a, i)
            i_index = np.delete(i_index, i)
        if temp_c.shape in [(0, 1), (1, 0)]:
            break
    return X, abs(np.sum(X * c))


def check_optimal(c, x, m, n):
    i_index, j_index = np.where(x > 0)
    A = np.zeros((i_index.shape[0] + 1, m + n))
    b = np.zeros(i_index.shape[0] + 1)
    A[0][0] = 1
    for i in range(i_index.shape[0]):
        A[i + 1][i_index[i]] = 1
        A[i + 1][m + j_index[i]] = 1
        b[i + 1] = c[i_index[i]][j_index[i]]
    x = np.linalg.solve(A, b)
    u = x[:m]
    v = x[m:]

    P = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            P[i][j] = c[i][j] - (u[i] + v[j])
    return P


def u_v(m, n, c, a, b, X):
    P = check_optimal(c, X, m, n)
    while np.min(P) < 0:
        i, j = np.argmin(P) // n, np.argmin(P) % n
        X[i][j] = 1
        use_row = True
        queue = [(i, j)]
        is_add = [True]
        while True:
            temp_i = queue[-1][0]
            temp_j = queue[-1][1]
            # Row
            if use_row:
                if temp_i == i and temp_j != j:
                    break
                positive_indices = np.where(X[temp_i] > 0)  # old basic
                if positive_indices[0].shape[0] == 1:
                    queue.pop()
                    is_add.pop()
                    continue
                else:
                    del_pointer = -2
                    while len(queue) > 2:
                        if queue[del_pointer - 1][1] != queue[-1][1]:
                            break
                        queue.pop(del_pointer)
                        is_add.pop(del_pointer)
                    cur_qi = len(queue) - 1
                    for pi in positive_indices[0]:
                        if pi == temp_j:
                            continue
                        queue.append((temp_i, pi))
                        is_add.append(not is_add[cur_qi])
            else:
                # Col
                if temp_i != i and temp_j == j:
                    break
                positive_indices = np.where(X[:, temp_j] > 0)  # old basic
                if positive_indices[0].shape[0] == 1:
                    queue.pop()
                    is_add.pop()
                    continue
                else:
                    del_pointer = -2
                    while len(queue) > 2:
                        if queue[del_pointer - 1][0] != queue[-1][0]:
                            break
                        queue.pop(del_pointer)
                        is_add.pop(del_pointer)
                    cur_qi = len(queue) - 1
                    for pi in positive_indices[0]:
                        if pi == temp_i:
                            continue
                        queue.append((pi, temp_j))
                        is_add.append(not is_add[cur_qi])

            use_row = not use_row
        X[i][j] = 0
        theta = None
        for qi in range(len(queue)):
            if not is_add[qi]:
                temp = X[queue[qi][0]][queue[qi][1]]
                theta = temp if (theta is None or theta > temp) else theta
        for qi in range(len(queue)):
            if is_add[qi]:
                X[queue[qi][0]][queue[qi][1]] += theta
            else:
                X[queue[qi][0]][queue[qi][1]] -= theta
        P = check_optimal(c, X, m, n)
    return X, abs(np.sum(X * c))


def lp(m, n, c, a, b):
    m, n, c, a, b = balance(m, n, c, a, b)
    X, total_cost = least_cost_cell(m, n, c, a, b)

    X, total_cost = u_v(m, n, c, a, b, X)
    return X, total_cost


def getResutl(m, n, c, a, b):
    X, total_cost = lp(m, n, c, a, b)

    return X, total_cost
