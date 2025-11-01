import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from astropy.io import fits
import scipy.stats as stats
from skimage.transform import resize
from skimage.io import imread
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
from PIL import Image, ImageTk

# Default Constants
DEFAULTS = {
    "Pixel Size (um)": 3.76e-6,
    "Sensor Width (px)": 9568,
    "Sensor Height (px)": 6380,
    "Aperture": 0.111,
    "QE": 0.6,
    "Wavelength (nm)": 640,
    "Dark Current (e-)": 0.56,
    "Saturation Capacity (e-)": 51000,
    "Readout Noise (e-)": 1,
    "Width Field of View (deg)": 10,
    "Max Magnitude": 20,
    "Min Magnitude": 12,
    "Zero Point": 18,
    "PSF (sigma)": 3,
    "Exposure Time": 10,
    "Num of Stars": 1000
}

OPTIONAL_PARAMS = {
    "Trail Length (px)": 10,
    "Drift Angle (deg)": 0,
    "Cosmic Ray Count": 5,
    "Cosmic Ray Max Length": 20,
    "Cosmic Ray Intensity (e-)": 5000,
    "Sky Background Rate (e-/px/s)": 0.1,
    "Image Scale Factor": 1.0,
    "Image Magnitude": 1.0
}

## Helper Functions

def apply_binning(image, bin_size=3):
    """Bin the image in non-overlapping blocks of bin_size x bin_size pixels."""
    h, w = image.shape
    # Trim the image so that dimensions are divisible by bin_size:
    h_trim = h - (h % bin_size)
    w_trim = w - (w % bin_size)
    trimmed = image[:h_trim, :w_trim]
    # Reshape and sum over the binning blocks.
    binned = trimmed.reshape(h_trim // bin_size, bin_size, w_trim // bin_size, bin_size).sum(axis=(1,3))
    return binned

def add_cosmic_rays(image, num_rays=5, max_length=20, intensity=5000):
    ## Add simulated cosmic ray events as short bright streaks."""
    for _ in range(num_rays):
        # Choose a random starting pixel.
        start_x = np.random.randint(0, image.shape[1])
        start_y = np.random.randint(0, image.shape[0])
        # Random length and direction.
        length = np.random.randint(1, max_length)
        angle = np.random.uniform(0, 2 * np.pi)
        for i in range(length):
            x = int(start_x + i * np.cos(angle))
            y = int(start_y + i * np.sin(angle))
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                image[y, x] += intensity
    return image

def add_sky_background(image, background_rate, exposure_time):
    """Add a uniform sky background (in electrons) across the image."""
    # In a more detailed model, you’d need the sun’s elevation, atmospheric conditions, etc.
    # Check skybackcalc.py for a (presumably) more accurate calculation.
    return image + background_rate * exposure_time

imported_image = []
image_preview_canvas = None

def import_image():
    global imported_image
    filename = filedialog.askopenfilename(
        title="Select a PNG, JPEG, or FITS File",
        filetypes=[
            ("PNG Files", "*.png"),
            ("FITS Files", "*.fits"),
            ("JPG Files", "*.jpg"),
            ("JPEG Files", "*.jpeg"),
            ("All Files", "*.*"),
        ]
    )

    if not filename:  # If user cancels, exit function safely
        return

    try:
        if filename.lower().endswith(".fits"):
            with fits.open(filename) as hdul:
                imported_image = hdul[0].data.astype(float)
        else:
            imported_image = imread(filename, as_gray=True).astype(float)

        if imported_image is None or imported_image.size == 0:
            raise ValueError("Loaded image is empty or invalid.")

        messagebox.showinfo("Success", "Image imported successfully!")
        update_image_preview()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {str(e)}")
        imported_image = None  # Reset in case of failure

def update_image_preview():
    global image_preview_canvas, imported_image
    if imported_image is not None:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(imported_image, cmap='gray', origin='lower')
        ax.set_title("Imported Image Preview")
        ax.axis('off')
        
        if image_preview_canvas:
            image_preview_canvas.get_tk_widget().destroy()
        
        image_preview_canvas = FigureCanvasTkAgg(fig, master=preview_frame)
        image_preview_canvas.get_tk_widget().pack()
        image_preview_canvas.draw()

def calculate_image_flux(image_magnitude, sensor_params):
    """Calculate appropriate photon flux per pixel for the imported image."""
    total_flux = 10 ** (-0.4 * (image_magnitude - sensor_params["Zero Point"]))
    total_photons = total_flux * sensor_params["Exposure Time"]  # Adjust based on exposure time
    total_photons *= sensor_params["QE"]  # Adjust based on quantum efficiency
    # total_photons *= 0.1 # Adjust based on \delta 30 nm bandpass filter
    pixel_flux = total_photons / np.sum(imported_image)  # Normalize across image pixels
    return pixel_flux

def add_imported_image(image, scale_factor=1.0, image_magnitude=10.0, sensor_params=None):
    global imported_image
    if imported_image is not None:
        original_h, original_w = imported_image.shape
        sensor_h, sensor_w = image.shape

        # Compute new dimensions while maintaining aspect ratio
        new_width = int(sensor_w * scale_factor)
        new_height = int(original_h * (new_width / original_w))

        # Ensure the resized image does not exceed sensor dimensions
        new_height = min(new_height, sensor_h)
        new_width = min(new_width, sensor_w)

        resized_image = resize(imported_image, (new_height, new_width), anti_aliasing=True)
        pixel_flux = calculate_image_flux(image_magnitude, sensor_params)
        resized_image *= pixel_flux  # Scale the image based on calculated flux

        # Compute centering positions
        start_x = max(0, (sensor_w - new_width) // 2)
        start_y = max(0, (sensor_h - new_height) // 2)

        # Ensure the final cropped region does not exceed bounds
        end_x = start_x + new_width
        end_y = start_y + new_height

        # Crop to ensure compatibility with NumPy broadcasting
        image[start_y:end_y, start_x:end_x] += resized_image[:end_y - start_y, :end_x - start_x]

    return image

### PSF addition ###

def add_psf(image, x, y, flux, sigma):
    """Add a 2D Gaussian PSF to the image at (x, y) with the given flux."""
    size = int(6 * sigma)
    # Create grid centered on 0
    y_indices, x_indices = np.meshgrid(np.arange(-size//2, size//2+1),
                                       np.arange(-size//2, size//2+1), indexing='ij')
    psf = np.exp(-(x_indices**2 + y_indices**2) / (2 * sigma**2))
    psf /= psf.sum()
    ix, iy = int(y), int(x)
    # Determine sub-image boundaries
    if 0 <= ix < image.shape[0] and 0 <= iy < image.shape[1]:
        x_start, x_end = max(0, ix - size//2), min(image.shape[0], ix + size//2+1)
        y_start, y_end = max(0, iy - size//2), min(image.shape[1], iy + size//2+1)
        sub_psf = psf[:x_end-x_start, :y_end-y_start]
        image[x_start:x_end, y_start:y_end] += flux * sub_psf

# Starfield search and generation functions

def query_realistic_starfield(ra, dec, fov_deg_h, fov_deg_w, max_mag_limit, min_mag_limit):
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    # radius = (fov_deg / 2) * u.deg
    vizier = Vizier(columns=['RAJ2000', 'DEJ2000', 'magG'])
    vizier.ROW_LIMIT = -1
    result = vizier.query_region(
        coord, height=fov_deg_h * u.deg, width=fov_deg_w * u.deg, catalog='igsl3', column_filters={'magG': f'< {max_mag_limit}', 'magG': f'> {min_mag_limit}'})
    if result:
        stars = result[0]
        return stars
    else:
        messagebox.showerror("Error", "No stars found in the specified field of view.")
        return None

def generate_realistic_star_positions(params, stars):
    sensor_h = int(params["Sensor Height (px)"])
    sensor_w = int(params["Sensor Width (px)"])
    fov_deg = params["Width Field of View (deg)"]
    pixel_scale = fov_deg / sensor_w  # degrees per pixel

    center_ra = float(ra_entry.get())
    center_dec = float(dec_entry.get())

    x_positions = []
    y_positions = []
    magnitudes = []

    print(f"Generating positions for {len(stars)} stars within RA {center_ra} and Dec {center_dec} with width {fov_deg} degrees and height {fov_deg * (sensor_h / sensor_w)} degrees.")
    for star in tqdm(stars):
        delta_ra = (star['RAJ2000'] - center_ra)
        delta_dec = (star['DEJ2000'] - center_dec)

        x = sensor_w / 2 + delta_ra / pixel_scale
        y = sensor_h / 2 + delta_dec / pixel_scale

        if 0 <= x < sensor_w and 0 <= y < sensor_h:
            # print(f"Star within bound: ({x:.2f}, {y:.2f})")
            x_positions.append(x)
            y_positions.append(y)
            magnitudes.append(star['magG'])
            # print(magnitudes[-1], "magnitude at position (", x, ",", y, ")")
            # print(f"Star at ({x :.2f}, {y:.2f}) with magnitude {star['magG']:.2f}")
    return x_positions, y_positions, magnitudes

### Main image generation function ###

def generate_random_star_positions(params):
    """Generate random star positions and magnitudes within the sensor dimensions."""
    sensor_h = int(params["Sensor Height (px)"])
    sensor_w = int(params["Sensor Width (px)"])
    x_positions = []
    y_positions = []
    magnitudes = []

    num_stars = int(params["Num of Stars"])
    x_positions = np.random.uniform(0, sensor_w, num_stars)
    y_positions = np.random.uniform(0, sensor_h, num_stars)
    magnitudes = np.random.uniform(params["Min Magnitude"], params["Max Magnitude"], num_stars)

    return x_positions, y_positions, magnitudes

def generate_image(params, binning=False, cosmic_rays=False, sky_background=False, moving_exposures=False, snr_calc=False, import_image=False, realistic=False, pbar=None, win=None, progress_update=None):

    image = np.zeros((int(params["Sensor Height (px)"]), int(params["Sensor Width (px)"])))
    if progress_update:
        progress_update(10, "Generating empty image...")

    if realistic_var.get():
        print("Generating realistic star field.")
        try:
            ra = float(ra_entry.get())
            dec = float(dec_entry.get())
            fov_height = params["Width Field of View (deg)"] * (params["Sensor Height (px)"] / params["Sensor Width (px)"])
            if progress_update:
                progress_update(10, "Querying star positions...")

            stars = query_realistic_starfield(ra, dec, fov_height, params["Width Field of View (deg)"], params["Max Magnitude"], params["Min Magnitude"])
            print("Stars found:", len(stars))
            if stars is None:
                raise ValueError("No stars found in field of view.")

            if progress_update:
                progress_update(10, "Populating star positions...")

            x_positions, y_positions, magnitudes = generate_realistic_star_positions(params, stars)

        except Exception as e:
            messagebox.showerror("Error", f"Realistic mode failed: {e}.")
    else:
        if progress_update:
           progress_update(20, "Generating random star positions...")
        x_positions, y_positions, magnitudes = generate_random_star_positions(params)
        # print(f"Generated {len(x_positions)} star positions.")
    
    if progress_update:
        progress_update(20, "Adding star PSFs to image...")

    if moving_exposures:
        # Divide the exposure into subexposures to simulate camera drift.
        num_steps = 10  # Can be a parameter?

        trail_length = params["Trail Length (px)"]
        drift_angle_rad = np.deg2rad(params["Drift Angle (deg)"])
        dx = trail_length * np.cos(drift_angle_rad) / (num_steps - 1)
        dy = trail_length * np.sin(drift_angle_rad) / (num_steps - 1)
        for step in range(num_steps):
            sub_exposure_time = params["Exposure Time"] / num_steps
            # For each subexposure, add the star signals with a small positional offset
            for x, y, mag in tqdm(zip(x_positions, y_positions, magnitudes), total=len(x_positions)):
                flux = 10 ** (-0.4 * (mag - params["Zero Point"]))
                photons = flux * sub_exposure_time
                electrons = np.random.poisson(photons * params["QE"])
                add_psf(image, x + dx * step, y + dy * step, electrons, params["PSF (sigma)"])
    else:
        # Normal (static) exposure
        for x, y, mag in tqdm(zip(x_positions, y_positions, magnitudes), total=len(x_positions)):
            flux = 10 ** (-0.4 * (mag - params["Zero Point"]))
            photons = flux * params["Exposure Time"]
            electrons = np.random.poisson(photons * params["QE"])
            add_psf(image, x, y, electrons, params["PSF (sigma)"])

    if progress_update:
        progress_update(20, "Adding noise to image...")

    # Add imported image if available
    if import_image:
        image = add_imported_image(image, params["Image Scale Factor"], params["Image Magnitude"], params)

    # Add cosmic rays if toggled
    if cosmic_rays:
        image = add_cosmic_rays(image,
                                 num_rays=int(params["Cosmic Ray Count"]),
                                 max_length=int(params["Cosmic Ray Max Length"]),
                                 intensity=int(params["Cosmic Ray Intensity (e-)"]))
    
    # Add sky background if toggled
    if sky_background:
        image = add_sky_background(image,
                                   background_rate=params["Sky Background Rate (e-/px/s)"],
                                   exposure_time=params["Exposure Time"])
    
    # Add dark noise and readout noise
    dark_noise = np.random.poisson(params["Dark Current (e-)"] * params["Exposure Time"], image.shape)
    readout_noise = np.random.normal(params["Readout Noise (e-)"], 1.5, image.shape).astype(int)
    image += dark_noise + readout_noise
    
    if progress_update:
            progress_update(20, "Finalizing image...")

    # Clip the image to sensor's saturation capacity
    image = np.clip(image, 0, params["Saturation Capacity (e-)"]).astype(int)
    
    # Apply 3x3 binning if toggled
    if binning:
        print('Applying 3x3 binning')
        image = apply_binning(image, bin_size=3)
    
    if snr_calc:
        print('Calculating SNR')
        # Calculate the signal-to-noise ratio.
        
        signal = image.mean()
        noise = np.std(image)
        snr = 10*np.log10(signal / noise)
        print(f"Signal: {signal}, Noise: {noise}, SNR: {snr}")

        # This is a simple estimate assuming Poisson noise for CCD/CMOS.
        signal = params["Exposure Time"] * params["QE"] * 10**(-0.4 * (params["Min Magnitude"] - params["Zero Point"]))
        noise = np.sqrt(signal + int(sky_background_var.get()) * params["Sky Background Rate (e-/px/s)"] + params["Dark Current (e-)"] * params["Exposure Time"] + params["Readout Noise (e-)"]**2)
        snr = signal / noise
        print("Expected Signal: ", signal, "Expected Noise: ", noise, "Expected SNR: ", snr)

    
    return image

### Image saving ###

def save_image(image, filename, format):
    if format == 'png':
        plt.imsave(filename, image, cmap='gray')
    elif format == 'fits':
        fits.writeto(filename, image, overwrite=True)

### GUI Control ###

def run_simulation():
    # Retrieve parameters from the text entries
    win, pbar, status_label = progress_window()

    def update_progress(value, message=None):
        pbar.step(value)
        if message:
            status_label.config(text=message)
            print(message)
        win.update()

    pbar['value'] = 0
    update_progress(0, "Fetching Parameters")
    win.update()

    params = {key: float(entries[key].get()) if entries[key].get() else DEFAULTS[key]
              for key in DEFAULTS}
    params.update({key: float(opt_entries[key].get()) if opt_entries[key].get() else OPTIONAL_PARAMS[key]
              for key in OPTIONAL_PARAMS})

    image = generate_image(params,
                           binning=binning_var.get(),
                           cosmic_rays=cosmic_rays_var.get(),
                           sky_background=sky_background_var.get(),
                           moving_exposures=moving_exposures_var.get(),
                           snr_calc=snr_calc_var.get(),
                           import_image=import_image_var.get(),
                           realistic=realistic_var.get(), progress_update=update_progress)
    
    update_progress(9.9, 'Loading image rendering')

    fig = plt.figure(facecolor=(0.2, 0.2, 0.2))
    # plt.style.use('dark_background')
    # fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(111)

    COLOR = 'white'
    mpl.rcParams['text.color'] = COLOR
    mpl.rcParams['axes.labelcolor'] = COLOR
    mpl.rcParams['xtick.color'] = COLOR
    mpl.rcParams['ytick.color'] = COLOR
    mpl.rcParams['axes.edgecolor'] = COLOR
    mpl.rcParams['figure.facecolor'] = (0.2, 0.2, 0.2)
    mpl.rcParams['figure.edgecolor'] = COLOR
    mpl.rcParams['font.family'] = 'monospace'

    ax.set_title("Simulated CMOS Image", color='white')
    ax.set_aspect('equal')

    ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
    spines = ['top', 'bottom', 'left', 'right']
    for spine in spines:
        ax.spines[spine].set_color('white')
        ax.spines[spine].set_linewidth(0.5)

    update_progress(0, "Displaying image...")

    Image = ax.imshow(image, cmap='gray', origin='lower')
    fig.subplots_adjust(bottom=0.2, left=0.1)
    fig.colorbar(Image, ax=ax, label='Electron Count', shrink=0.8)
    vmin = fig.add_axes([0.15, 0.15, 0.7, 0.02])
    vmax = fig.add_axes([0.15, 0.10, 0.7, 0.02])
    vmin_slider = Slider(vmin, 'Min Value', 0, image.max(), valinit=0, color='red')
    vmax_slider = Slider(vmax, 'Max Value', 0, image.max(), valinit=image.max(), color='red')
    
    def update(val):
        new_vmin = vmin_slider.val
        new_vmax = vmax_slider.val
        Image.set_clim(new_vmin, new_vmax)
        fig.canvas.draw_idle() # Redraw the figure
        plt.draw() # Update the canvas

    vmin_slider.on_changed(update)
    vmax_slider.on_changed(update)

    # fig.tight_layout()
    win.destroy()
    plt.show()

    # Plot histogram of pixel values
    # mu, sigma = np.mean(image), np.std(image)
    # x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    # plt.plot(x, 6*stats.norm.pdf(x, mu, sigma), linewidth=0.5, label='Gaussian Fit')
    # plt.hist(image.flatten(), bins=100, range=(0, image.max()))
    # plt.yscale('log')
    # plt.show()

def save_file():
    filename = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG Image", "*.png"), ("FITS File", "*.fits")])
    if filename:
        format = "fits" if filename.endswith(".fits") else "png"
        params = {key: float(entries[key].get()) if entries[key].get() else DEFAULTS[key]
                  for key in DEFAULTS}
        # Retrieve optional parameters
        params.update({key: float(entries[key].get()) if entries[key].get() else OPTIONAL_PARAMS[key]
                       for key in OPTIONAL_PARAMS})
        image = generate_image(params,
                               binning=binning_var.get(),
                               cosmic_rays=cosmic_rays_var.get(),
                               sky_background=sky_background_var.get(),
                               moving_exposures=moving_exposures_var.get(),
                               snr_calc = snr_calc_var.get(),
                               import_image = import_image_var.get(),
                               realistic = realistic_var.get())
        save_image(image, filename, format)
        messagebox.showinfo("Success", f"Image saved as {filename}")

def progress_window():
    new_window = tk.Toplevel(root)
    new_window.title("Simulation Progress")
    # new_window.iconbitmap("NewDLogo.ico")

    tk.Label(new_window, text="Simulation in Progress...", font=("TkDefaultFont", 24, 'bold')).grid(row=0, column=0, padx=100, pady=20, sticky='sew')
    status_label = tk.Label(new_window, text="", font=("TkDefaultFont", 20))
    status_label.grid(row=1, column=0, padx=100, sticky='ew')
    # status_label.pack(pady=5)
    
    new_window.geometry("+100+100")
    new_window.resizable(False, False)

    progress_bar = ttk.Progressbar(new_window, mode="determinate", length=300, orient="horizontal", maximum=100)
    progress_bar.grid(row=2, column=0, padx=50, pady=20,sticky='new')
    return new_window, progress_bar, status_label



### GUI Setup ###

root = tk.Tk()
root.title("CMOS Image Simulation GUI")

title_frame = tk.Frame(root)
title_frame.grid(row=0, column=0, columnspan=2, padx=120, pady=10, sticky="nsew")
tk.Label(title_frame, text="CMOS Image Simulation", font=("TkDefaultFont", 24, 'bold')).grid(row=0, column=0, sticky="w")
tk.Label(title_frame, text="by WashU Satellite", font=("TkDefaultFont", 20)).grid(row=1, column=0, sticky="w")
tk.Label(title_frame, text="Version 1.0", font=("TkDefaultFont", 16)).grid(row=2, column=0, sticky="w")

logo_frame = tk.Frame(root)
logo_frame.grid(row=0, column=0, padx=20, pady=10, sticky="nw")
logo_img = Image.open("cmos-sims/NewDLogo.png")
title_frame.update_idletasks()
logo_img = logo_img.resize((title_frame.winfo_height(), title_frame.winfo_height()), Image.LANCZOS)
logo = ImageTk.PhotoImage(logo_img)
logo_label = tk.Label(logo_frame, image=logo)
logo_label.pack()
logo_label.image = logo # reference for garbage collection

realistic_var = tk.BooleanVar(value=False)
ra_entry = tk.StringVar()
dec_entry = tk.StringVar()

left_frame = tk.Frame(root)
left_frame.grid(row=1, column=0, padx=10, sticky="nsew")

# Create parameter entries frame
param_frame = tk.LabelFrame(left_frame, text="Default Parameters", padx=10, pady=10)
param_frame.grid(row=0, column=0, padx=10, sticky="new")

entries = {}
# List default parameters in GUI
for i, (key, value) in enumerate(DEFAULTS.items()):
    tk.Label(param_frame, text=key).grid(row=i, column=0, pady=4, sticky="w")
    entries[key] = tk.Entry(param_frame, width=12)
    entries[key].grid(row=i, column=1)
    entries[key].insert(0, str(value))

button_frame = tk.Frame(root)
button_frame.grid(row=1, column=1, sticky="nsew")

# GUI widgets for Realistic Mode
realistic_toggle_frame = tk.LabelFrame(button_frame, text="Star Field Mode", padx=10, pady=10)
realistic_toggle_frame.grid(row=0, column=0, padx=10, sticky="nsew")

mode_label = tk.Label(realistic_toggle_frame, text="Use Realistic Star Field:")
mode_label.grid(row=0, column=0, sticky="w")
tk.Checkbutton(realistic_toggle_frame, variable=realistic_var).grid(row=0, column=1, sticky="w")

tk.Label(realistic_toggle_frame, text="RA (deg):").grid(row=1, column=0, sticky="w")
tk.Entry(realistic_toggle_frame, textvariable=ra_entry, width=10).grid(row=1, column=1, sticky="w")

tk.Label(realistic_toggle_frame, text="Dec (deg):").grid(row=2, column=0, sticky="w")
tk.Entry(realistic_toggle_frame, textvariable=dec_entry, width=10).grid(row=2, column=1, sticky="w")

# GUI for optional toggles
toggle_frame = tk.LabelFrame(button_frame, text="Optional Features", padx=10, pady=10)
toggle_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

# opt_vals = tk.Frame(toggle_frame).grid(row=0, column=0)
# opt_bools = tk.Frame(toggle_frame).grid(row=0, column=1)

binning_var = tk.BooleanVar(value=False)
cosmic_rays_var = tk.BooleanVar(value=False)
sky_background_var = tk.BooleanVar(value=False)
moving_exposures_var = tk.BooleanVar(value=False)
snr_calc_var = tk.BooleanVar(value=False)
import_image_var = tk.BooleanVar(value=False)

tk.Checkbutton(toggle_frame, text="Simulate Moving Exposures", variable=moving_exposures_var).grid(row=0, column=0, sticky='w')
tk.Checkbutton(toggle_frame, text="Add Cosmic Rays", variable=cosmic_rays_var).grid(row=1, column=0, sticky='w')
tk.Checkbutton(toggle_frame, text="Add Sky Background", variable=sky_background_var).grid(row=2, column=0, sticky='w')
tk.Checkbutton(toggle_frame, text="Import Image", variable=import_image_var).grid(row=3, column=0, sticky='w')
tk.Checkbutton(toggle_frame, text="3x3 Binning", variable=binning_var).grid(row=1, column=1, sticky='w')
tk.Checkbutton(toggle_frame, text="Calculate SNR", variable=snr_calc_var).grid(row=2, column=1, sticky='w')

opt_param_frame = tk.LabelFrame(button_frame, text="Optional Parameters", padx=10, pady=10)
opt_param_frame.grid(row=2, column=0, padx=10, sticky="nsew")

opt_entries = {}
for i, (key, value) in enumerate(OPTIONAL_PARAMS.items()):
    tk.Label(opt_param_frame, text=key).grid(row=i, column=0, sticky="w")
    opt_entries[key] = tk.Entry(opt_param_frame, width=12)
    opt_entries[key].grid(row=i, column=1)
    opt_entries[key].insert(0, str(value))

toggle_map = [
    (cosmic_rays_var, ["Cosmic Ray Count", "Cosmic Ray Max Length", "Cosmic Ray Intensity (e-)"]),
    (sky_background_var, ["Sky Background Rate (e-/px/s)"]),
    (import_image_var, ["Image Scale Factor", "Image Magnitude"]),
    (moving_exposures_var, ["Trail Length (px)", "Drift Angle (deg)"]),
]

def toggle_entries(var, keys):
    state = "normal" if var.get() else "disabled"
    for key in keys:
        opt_entries[key].config(state=state)

for var, keys in toggle_map:
    var.trace_add("write",
        lambda *_, v=var, ks=keys: toggle_entries(v, ks)
    )
    toggle_entries(var, keys)

for e in opt_entries.values():
    e.config(disabledbackground="#969595",
             disabledforeground="#666666",
             relief="flat",
             bd=1,)

# Create frame for action buttons
button_frame = tk.Frame(root)
button_frame.grid(row=3, column=0, columnspan=2, padx=20, pady=20)
tk.Button(button_frame, text="Run Simulation", command=run_simulation, width=20, height=2).grid(row=0, column=0, padx=5, sticky="nsew")
tk.Button(button_frame, text="Save Image", command=save_file, width=20, height=2).grid(row=0, column=1, padx=5, sticky="nsew")
tk.Button(button_frame, text="Import Image", command=import_image, width=20, height=2).grid(row=0, column=2, padx=5, sticky="nsew")

global current_progress
current_progress = 'Initializing...'

preview_frame = tk.LabelFrame(root, text="Imported Image Preview", padx=10, pady=10)
preview_frame.grid(row=1, column=2, padx=10, pady=10, sticky="n")
    
root.mainloop()