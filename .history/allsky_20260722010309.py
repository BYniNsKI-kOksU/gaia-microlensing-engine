    extent = (-np.pi, np.pi, -np.pi/2, np.pi/2)
    ax.imshow(
        hist,
        extent=extent,
        origin="lower",
        cmap=cmap_mw,
        interpolation="nearest",
        aspect="auto",
        zorder=0,
    )