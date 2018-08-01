class Softmax():
    def __call__(self, z):
        shifted = z - z.max()
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def gradient(self,Y):
        value = self(Y)
        J = -value[..., None] * value[:, None, :]
        iy, ix = np.diag_indices_from(J[0])
        J[:, iy, ix] = value * (1 - value)
        print(J.shape)
        return J
