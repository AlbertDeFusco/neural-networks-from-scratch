import numpy as np
import matplotlib.pyplot as plt

def decision_boundary(model, X, Y, ax=None):
    # Plot the decision boundary
    # Determine grid range in x and y directions
    x_min, x_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
    y_min, y_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1

    # Set grid spacing parameter
    spacing = min(x_max - x_min, y_max - y_min) / 500

    # Create grid
    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                   np.arange(y_min, y_max, spacing))

    # Concatenate data to match input
    data = np.hstack((XX.ravel().reshape(-1,1), 
                      YY.ravel().reshape(-1,1)))
    # Get decision boundary probabilities
    try:
        clf = model.predict_classes(data)
    except AttributeError:
        clf = model.predict(data)

    Z = clf.reshape(XX.shape)

    if ax is None:
        fig, ax = plt.subplots()
    ax.contourf(XX, YY, Z, cmap=plt.cm.Spectral, alpha=0.8)
    ax.scatter(X[:,0], X[:,1], c=Y,
                cmap=plt.cm.Spectral)
    return ax
