MAP_FILE = "gaia_allsky_hammer_16k.png"


def create_background_map():
    # Funkcja pozostaje, ale nie tworzy pustej mapy, ponieważ używamy gotowego obrazu tła.
    pass


# W części renderowania animacji, zmień kolor scatter dla zdarzeń mikrosoczewkowania na czerwony:
scatter = ax.scatter(
    x_proj,
    y_proj,
    s=sizes,
    c="red",  # zmieniono z "black" na "red"
    alpha=0.7,
    edgecolors="none",
)