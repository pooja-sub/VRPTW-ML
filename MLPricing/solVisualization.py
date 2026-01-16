import matplotlib
from matplotlib import pyplot as plt
from route import Route
from paramsVRP import ParamsVRP

matplotlib.use('TkAgg')  # Or try 'Agg'

def solVis(user_param, routes, sol_time, opt_cost, dataset_name, POPOUT=False):
    """
    Visualize the solution of the vehicle routing problem.
    """
    # Get customer locations
    depot_x, depot_y = user_param.posx[0], user_param.posy[0]
    customer_x = user_param.posx[1:-1]
    customer_y = user_param.posy[1:-1]

    # Create figure
    plt.figure(figsize=(10, 8))
    plt.title(f"B&P Solution for VRPTW on dataset {dataset_name}", fontsize=16)

    # Plot depot
    plt.scatter(depot_x, depot_y, color="black", s=100, label="Depot", zorder=5)
    plt.text(depot_x, depot_y, "Depot", fontsize=12, ha="right", va="bottom", color="black")

    # Plot customer points
    plt.scatter(customer_x, customer_y, color="gray", s=50, label="Customers", zorder=5)
    for i in range(len(customer_x)):
        plt.text(customer_x[i], customer_y[i], f"{i + 1}", fontsize=10, ha="right", va="bottom", color="gray")

    # Plot routes
    cmap = plt.cm.tab20  # Use academic journal recommended colormap
    for i, route in enumerate(routes):
        path = route.get_path()
        route_x = [user_param.posx[j] for j in path]
        route_y = [user_param.posy[j] for j in path]
        color = cmap(i % cmap.N)  # Cycle through colormap colors
        plt.plot(route_x, route_y, color=color, linewidth=2, label=f"Route {i + 1}", zorder=1)
        for j in range(len(path) - 1):
            start_x, start_y = route_x[j], route_y[j]
            end_x, end_y = route_x[j + 1], route_y[j + 1]
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2

            # Calculate arrow direction
            dx = end_x - start_x
            dy = end_y - start_y

            # Add arrow
            plt.annotate("", xy=(mid_x, mid_y), xytext=(start_x, start_y),
                         arrowprops=dict(arrowstyle="->", color=color, lw=1.5))


    # Set legend
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("X Coordinate", fontsize=14)
    plt.ylabel("Y Coordinate", fontsize=14)
    plt.tight_layout()

    # Add solution information below the legend (lower right corner)
    info_text = (
        f"Optimal Cost: {opt_cost:.2f}\n"
        f"Total Time: {sol_time:.2f} seconds\n"
        #f"Routes: {routes.__str__()}"
    )
    plt.text(0.95, 0.1, info_text, fontsize=12, ha="left", va="top", transform=plt.gca().transAxes,
             bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"))

    # Display the figure
    filename = f"C:\\Users\\CiSTUP\\Desktop\\A-Branch-and-Price-Algorithm-for-VRPTW\\fig\\VRPTW_B&P_Sol_Dataset{dataset_name}.svg"
    plt.savefig(filename)
    if POPOUT:
        plt.show()
        plt.close()
    if not POPOUT:
        plt.close()

