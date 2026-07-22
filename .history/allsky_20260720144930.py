# Jednolite wygładzenie całej mapy
hist = hist.T
hist = gaussian_filter(hist, sigma=2.0)