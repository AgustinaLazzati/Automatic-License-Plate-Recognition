"""
This is the pipeline for the introduction to data properties exploration. 

Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres"
__license__ = "GPL"
__email__ = "debora,gtorres@cvc.uab.es"

"""

##### PYTHON PACKAGES
# import the necessary packages
import numpy as np
import cv2
import glob
import os
from imutils import perspective
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt

# OWN FUNCTIONS (MODIFY ACORDING TO YOUR LOCAL PATH)
from DataExploration.LicensePlateDetector import detectPlates


"""
#### EXP-SET UP
# DB Main Folder (MODIFY ACCORDING TO YOUR LOCAL PATH)
DataDir=r'data/Patentes'
Views=['FrontalAugmented','LateralAugmented']
"""

# Set up plots folder (overwrite)
PLOTS_DIR = "plots"

def setup_plots_folder():
    #Ensure the plots folder exists. We use an overwrite strategy:
    #- If files with the same names already exist they will be overwritten by plt.savefig(...)
    # We do NOT delete the folder or its contents to avoid permission/locking issues on Windows.
    os.makedirs(PLOTS_DIR, exist_ok=True)



#### COMPUTE PROPERTIES FOR EACH VIEW
def computeProperties (DataDir,Views):
    plateArea={}
    plateAngle={}
    imageColor={}
    imageIlluminance={}
    imageSaturation={}

    for View in Views:

        ImageFiles=sorted(glob.glob(os.path.join(DataDir,View,'*.jpg')))
        plateArea[View]=[]
        plateAngle[View]=[]
        imageColor[View]=[]
        imageIlluminance[View]=[]
        imageSaturation[View]=[]
        # loop over the images
        for imagePath in ImageFiles:
            # load the image
            image = cv2.imread(imagePath)
            # Image Color and Illuminance properties
            imageColor[View].append(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,0].flatten()))
            imageIlluminance[View].append(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,2].flatten()))
            imageSaturation[View].append(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,1].flatten()))
            # Image ViewPoint (orientation with respect frontal view and focal distance)
            regions, _ =detectPlates(image)  #OUR detectPlates RETURNS (regions, image)
            for reg in regions:
                # Region Properties
                reg = np.array(reg, dtype=np.float32)   # ensuring type is correct
                rect = cv2.minAreaRect(reg)
                
                plateArea[View].append(np.prod(rect[1]))
                # Due to the way cv2.minAreaRect computes the sides of the rectangle
                # (https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/)
                # Depending on view point, the estimated rectangle has not
                # the largest side along the horizontal axis. This cases are corrected to ensure that orientations
                # are always with respect the largest side 
                if (rect[1][0]<rect[1][1]):
                    plateAngle[View].append(rect[2]-90)
                else:
                    plateAngle[View].append(rect[2])

    return plateArea, plateAngle, imageColor, imageIlluminance, imageSaturation            



#### VISUALLY EXPLORE PROPERTIES DISTRIBUTION FOR EACH VIEW
# We now save plots to disk instead of showing them interactively.
def styled_boxplot(data_list, labels, box_colors, title, ylabel, save_path=None, show_plot=False):
    """
    Create a styled boxplot in the Insights style:
    - Colored boxes (semi-transparent)
    - Median line thick and black
    - Whiskers & caps dashed
    - Outliers colored like the box
    """
    plt.figure(figsize=(8,6))
    bp = plt.boxplot(data_list, patch_artist=True, tick_labels=labels, showfliers=True)

    # Style boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Style medians
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    # Style whiskers and caps as dashed lines
    for whisker in bp['whiskers']:
        whisker.set_linestyle('--')
        whisker.set_color('black')
        whisker.set_linewidth(1)
    for cap in bp['caps']:
        cap.set_linestyle('--')
        cap.set_color('black')
        cap.set_linewidth(1)

    # Style outliers
    for i, flier in enumerate(bp['fliers']):
        flier.set(marker='o', color=box_colors[i], alpha=1, markersize=6)

    plt.title(title)
    plt.ylabel(ylabel)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    elif show_plot:
        plt.show()
    else:
        plt.close()


def Visualplots(dataset_name, Views, plateAngle, imageColor, 
                imageIlluminance, imageSaturation, SHOW_SEPARATE=False, save_dir=PLOTS_DIR):

    co=['b','c']  # colors for views
    alpha_val = 0.7  # transparency

    def plot_hist_with_line(data_dict, title, filename, xlabel):
        plt.figure()
        for k, view in enumerate(Views):
            # Plot histogram
            counts, bins, _ = plt.hist(data_dict[view], bins=20, color=co[k], alpha=alpha_val, label=view)
            
            # KDE / smoothed distribution line scaled to histogram counts
            if len(data_dict[view]) > 1:
                kde = gaussian_kde(data_dict[view])
                x_vals = np.linspace(min(data_dict[view]), max(data_dict[view]), 200)
                # Scale KDE to match histogram
                kde_scaled = kde(x_vals) * len(data_dict[view]) * (bins[1] - bins[0])
                plt.plot(x_vals, kde_scaled, color=co[k], linewidth=2)
        plt.title(f"{dataset_name} {title} (Combined)")
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()


    # -------------------------------------------------------------
    # Color
    if SHOW_SEPARATE:
        for k, view in enumerate(Views):
            plt.figure()
            plt.hist(imageColor[view], bins=20, color=co[k], alpha=alpha_val)
            plt.title(f'{dataset_name} Color Distribution - {view}')
            plt.savefig(os.path.join(save_dir, f"{dataset_name}_Color_{view}.png"))
            plt.close()
    else:
        plot_hist_with_line(imageColor, "Color Distribution", f"{dataset_name}_Color_Combined.png", "Hue")

    # Boxplot
    x=[imageColor[v] for v in Views]
    styled_boxplot(
        data_list=x,
        labels=Views,
        box_colors=co,
        title=f'{dataset_name} Color Distribution',
        ylabel='Hue',
        save_path=os.path.join(save_dir, f"{dataset_name}_Color_Boxplot.png"),
        show_plot=SHOW_SEPARATE
    )

    # -------------------------------------------------------------
    # Saturation
    if SHOW_SEPARATE:
        for k, view in enumerate(Views):
            plt.figure()
            plt.hist(imageSaturation[view], bins=20, color=co[k], alpha=alpha_val)
            plt.title(f'Saturation Distribution - {view}')
            plt.savefig(os.path.join(save_dir, f"{dataset_name}_Saturation_{view}.png"))
            plt.close()
    else:
        plot_hist_with_line(imageSaturation, "Saturation Distribution", f"{dataset_name}_Saturation_Combined.png", "Saturation")

    # Boxplot
    x=[imageSaturation[v] for v in Views]
    styled_boxplot(
        data_list=x,
        labels=Views,
        box_colors=co,
        title=f'{dataset_name} Saturation Distribution',
        ylabel='Saturation',
        save_path=os.path.join(save_dir, f"{dataset_name}_Saturation_Boxplot.png"),
        show_plot=SHOW_SEPARATE
    )

    # -------------------------------------------------------------
    # Brightness
    if SHOW_SEPARATE:
        for k, view in enumerate(Views):
            plt.figure()
            plt.hist(imageIlluminance[view], bins=20, color=co[k], alpha=alpha_val)
            plt.title(f'{dataset_name} Brightness Distribution - {view}')
            plt.savefig(os.path.join(save_dir, f"{dataset_name}_Brightness_{view}.png"))
            plt.close()
    else:
        plot_hist_with_line(imageIlluminance, "Brightness Distribution", f"{dataset_name}_Brightness_Combined.png", "Brightness (V)")

    # Boxplot
    x=[imageIlluminance[v] for v in Views]
    styled_boxplot(
        data_list=x,
        labels=Views,
        box_colors=co,
        title=f'{dataset_name} Brightness Distribution',
        ylabel='Brightness (V)',
        save_path=os.path.join(save_dir, f"{dataset_name}_Brightness_Boxplot.png"),
        show_plot=SHOW_SEPARATE
    )

    # -------------------------------------------------------------
    # Plate Angle / Viewpoint
    if SHOW_SEPARATE:
        for k, view in enumerate(Views):
            plt.figure()
            plt.hist(plateAngle[view], bins=20, color=co[k], alpha=alpha_val)
            plt.title(f'{dataset_name} View Point Distribution - {view}')
            plt.savefig(os.path.join(save_dir, f"{dataset_name}_ViewPoint_{view}.png"))
            plt.close()
    else:
        plot_hist_with_line(plateAngle, "View Point Distribution", f"{dataset_name}_ViewPoint_Combined.png", "Angle (degrees)")

    # Boxplot
    x=[plateAngle[v] for v in Views]
    styled_boxplot(
        data_list=x,
        labels=Views,
        box_colors=co,
        title=f'{dataset_name} View Point Distribution',
        ylabel='Angle (degrees)',
        save_path=os.path.join(save_dir, f"{dataset_name}_ViewPoint_Boxplot.png"),
        show_plot=SHOW_SEPARATE
    )



# With this functions, we are going to translate the Hue values to be interpretable for humans 
# We will detect the colors in the images 
def hsv_to_color(h, s, v):
    #Convert HSV values (OpenCV ranges) to a rough car color name
    # Handle grayscale first (low saturation)
    if s < 30:  
        if v < 50:
            return "Black"
        elif v > 200:
            return "White"
        else:
            return "Gray"

    # Otherwise, classify by hue  (BECAUSE IF NOT, THE WHITE OR BLACK WILL BE CLASSIFY AS YELLOW OR GREEN)
    # We convert hue (0-179 OpenCV) to rough color name
    if h < 10 or h >= 160:
        return "Red"
    elif h < 25:
        return "Orange"
    elif h < 34:
        return "Yellow"
    elif h < 85:
        return "Green"
    elif h < 125:
        return "Blue"
    elif h < 145:
        return "Purple"
    elif h < 160:
        return "Pink"
    else:
        return "Unknown"


def summarize_colors(imageColor, imageSaturation, imageIlluminance, Views, dataset_name, plot=True, save_dir=PLOTS_DIR):
    """Summarize car colors per dataset using existing dictionaries, with bar chart."""
    colors = []
    
    # Iterate through all images in all views
    for v in Views:
        for i in range(len(imageColor[v])):
            h = imageColor[v][i]          # mean Hue
            s = imageSaturation[v][i]     # mean Saturation
            vval = imageIlluminance[v][i] # mean Value (brightness)
            colors.append(hsv_to_color(h, s, vval))
    
    # Count occurrences
    unique, counts = np.unique(colors, return_counts=True)
    color_counts = dict(zip(unique, counts))
    
    # Print summary
    print(f"\n{dataset_name} - Color Summary")
    for u, c in color_counts.items():
        print(f"  {u}: {c} images")
    
    expected = {"Red","Orange","Yellow","Green","Blue","Purple","Pink","Black","White","Gray"}
    missing = expected - set(unique)
    if missing:
        print(f"  Missing colors: {missing}")
    else:
        print("  All color categories present.")
    
    # Bar chart
    if plot:
        ordered_colors = ["Red","Orange","Yellow","Green","Blue","Purple","Pink","Black","White","Gray"]
        values = [color_counts.get(c, 0) for c in ordered_colors]
        
        plt.figure()
        color_map = {
            "Red": "red", "Orange": "orange", "Yellow": "yellow", "Green": "green",
            "Blue": "blue", "Purple": "purple", "Pink": "pink",
            "Black": "black", "White": "lightgray", "Gray": "gray"
        }
        bar_colors = [color_map[c] for c in ordered_colors]
        
        plt.bar(ordered_colors, values, color=bar_colors, edgecolor="k")
        plt.title(f"{dataset_name} - Car Colors")
        plt.ylabel("Number of Images")
        plt.xlabel("Color")
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(save_dir, f"{dataset_name}_CarColors.png"))
        plt.close()
    
    return color_counts

# Viewponts distribution
def compare_plate_angles(datasets, views, dataset_names, save_dir=PLOTS_DIR):
    #Compare the plate angles between datasets for each view.
    for view in views:
        print(f"\n--- Comparison for view: {view} ---")
        plt.figure()
        for plateAngle, name in zip(datasets, dataset_names):
            angles = plateAngle.get(view, [])
            if len(angles) == 0:
                continue
            mean_angle = np.mean(angles)
            std_angle = np.std(angles)
            print(f"{name}: mean={mean_angle:.2f}, std={std_angle:.2f}, n={len(angles)}")
            plt.hist(angles, bins=20, alpha=0.5, label=name)
        
        plt.title(f"Plate Angle Distribution - {view}")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"PlateAngles_{view}.png"))
        plt.close()



def main():
    # WE WILL COMPARE THREE DATASETS: real_plates, our own plates, and
    # an augmented version of our plates (modiffied properties).  
    SHOW_SEPARATE = False
    PLOT = True

    # Ensure plots folder exists (overwrite strategy)
    setup_plots_folder()

    # DB Main Folder (MODIFY ACCORDING TO YOUR LOCAL PATH)
    Real_DataDir=r'data'
    Views=['Frontal','Lateral']
    R_plateArea, R_plateAngle, R_imageColor, R_imageIlluminance, R_imageSaturation = computeProperties(Real_DataDir, Views)
    if PLOT:
       Visualplots('Real Data', Views, R_plateAngle, R_imageColor, R_imageIlluminance, R_imageSaturation)


    Own_DataDir=r'data/Patentes'
    O_plateArea, O_plateAngle, O_imageColor, O_imageIlluminance, O_imageSaturation = computeProperties(Own_DataDir, Views)
    if PLOT:
       Visualplots('Own Data', Views, O_plateAngle, O_imageColor, O_imageIlluminance, O_imageSaturation)
    
    #Augmented_DataDir=r'data/Patentes'
    #Views_A=['FrontalAugmented','LateralAugmented']
    #A_plateArea, A_plateAngle, A_imageColor, A_imageIlluminance, A_imageSaturation = computeProperties(Augmented_DataDir, Views_A)
    #if PLOT:
    #   Visualplots('Augmented Data', Views_A, A_plateAngle, A_imageColor, A_imageIlluminance, A_imageSaturation)

    # Interpreting what colors have the cars of our data set...
    summarize_colors(R_imageColor, R_imageSaturation, R_imageIlluminance, Views, "Real Data")
    summarize_colors(O_imageColor, O_imageSaturation, O_imageIlluminance, Views, "Own Data")
    #summarize_colors(A_imageColor, A_imageSaturation, A_imageIlluminance, Views_A, "Augmented Data")

    # Compare frontal and lateral views across datasets
    datasets = [R_plateAngle, O_plateAngle]
    dataset_names = ["Real Data", "Own Data"]
    views_to_compare = ["Frontal", "Lateral"]  # For augmented, adjust names if needed
    compare_plate_angles(datasets, views_to_compare, dataset_names)




"""
Explorar la distribucion de los histogramas para encontrar el protocolo.
crear un protocolo con el que se creo la imagen: distancia color, esas caracteristicas son tu protocolo. Es decir, hay una distancia de la foto a la patente. 
El protocolo lo define real_data, asi podes saber cuando la foto de entrada va a poder ser considerada. Definir entonces que imagenes se podran usar o no. 
despues con lo de los kernels quizas si lo cambias te va a cambiar el tamaño de las patentes, y todo eso lo tenemos que entender, compararlo eso. 

Dentro de los plots hacer lo del umbral/distribucion de los datos. por ejemplo el angulo o el zoom y ampliar la foto. despues de definir el protocolo, incluir o excluir fotos, volver 
a plotear con los nuevos. 

DISTRIBUCION Y ANALISIS DE DATOS. 

poner el hue entre 0 y 360
"""

if __name__ == "__main__":
    main()
