import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# Step 1: Simulate a dynamic data stream
def generate_dynamic_data_stream(size, anomaly_prob=0.01, anomaly_magnitude=10, noise_scale=0.5):
    """
    Simulate a continuous data stream, incorporating seasonal variations, random noise,
    and occasional anomalies (e.g., spikes, dips, shifts).

    Parameters:
    - size (int): The total number of data points to simulate.
    - anomaly_prob (float): Probability that any given point in the stream will be an anomaly.
    - anomaly_magnitude (float): Magnitude of anomalies (how large deviations are).
    - noise_scale (float): Scale of the random noise applied to the data.

    Returns:
    - A NumPy array representing the simulated data stream.
    
    This function generates a stream that mimics the behavior of real-world data sources (e.g., financial data,
    sensor readings). It includes a combination of deterministic patterns (seasonal variations, trends) and
    non-deterministic elements (noise, random anomalies).
    """
    # Generate a time index for the data
    time = np.arange(size)

    # Generate two seasonal components:
    # One short-term cycle (fast fluctuations) and one long-term cycle (slower, broader trends)
    seasonal_component = 5 * np.sin(2 * np.pi * time / 50) + 2 * np.sin(2 * np.pi * time / 150)

    # Add random noise to simulate small fluctuations and variability in real-world data
    noise = np.random.normal(0, noise_scale, size)

    # Simulate a trend (data slowly increases or decreases over time)
    trend = 0.01 * time

    # Combine all the components: baseline + seasonal patterns + noise + trend
    data = 10 + seasonal_component + noise + trend

    # Introduce anomalies (spikes, dips, or shifts) at random points in the data
    # Anomalies are important for testing how well the anomaly detection algorithm works
    anomalies_indices = np.random.choice(size, int(size * anomaly_prob), replace=False)
    anomaly_types = np.random.choice(['spike', 'dip', 'shift'], size=len(anomalies_indices))
    
    # Apply anomalies: A 'spike' increases value, a 'dip' decreases value, and a 'shift' modifies the trend over several points
    for i, anomaly_type in zip(anomalies_indices, anomaly_types):
        if anomaly_type == 'spike':
            data[i] += anomaly_magnitude
        elif anomaly_type == 'dip':
            data[i] -= anomaly_magnitude
        elif anomaly_type == 'shift':
            # Apply a shift over several consecutive points
            end_idx = min(i+5, size)
            data[i:end_idx] += anomaly_magnitude / 2 if random.random() > 0.5 else -anomaly_magnitude / 2

    return data

# Step 2: Calculate the Exponentially Weighted Moving Average (EWMA) and Z-score for anomaly detection
def ewma_z_score(new_value, history, avg, std_dev, alpha=0.2):
    """
    Use EWMA to smooth the data and Z-score to detect anomalies.

    Parameters:
    - new_value: The most recent data point to incorporate.
    - history (deque): A buffer of previous values, used to compute statistics.
    - avg (float): The average of the recent history.
    - std_dev (float): The standard deviation of the recent history.
    - alpha (float): The EWMA smoothing factor (controls how quickly it responds to new data).

    Returns:
    - ewma (float): The updated EWMA value for the current data point.
    - z_score (float): The Z-score, measuring how much the current data point deviates from the average.

    Explanation:
    EWMA is used for smoothing the data, making it easier to spot long-term trends and filter out short-term noise.
    Z-score measures how many standard deviations a data point is from the average, helping to flag outliers.
    """
    # Update the EWMA using the previous value and the new incoming data point
    ewma = (1 - alpha) * history[-1] + alpha * new_value if history else new_value

    # Calculate Z-score: deviation from the average, normalized by the standard deviation
    z_score = (new_value - avg) / std_dev if std_dev != 0 else 0
    return ewma, z_score

# Step 3: Detect anomalies in the data chunk using EWMA and Z-score
def detect_anomalies(data_chunk, adaptive_alpha=False):
    """
    Process a chunk of data and detect anomalies using EWMA and Z-score.

    Parameters:
    - data_chunk (array): The segment of data to process.
    - adaptive_alpha (bool): Whether to dynamically adjust the EWMA smoothing factor based on data volatility.

    Returns:
    - ewma_values (list): List of EWMA values for each data point in the chunk.
    - anomalies (list of tuples): Detected anomalies as (index, value) pairs.
    
    The function analyzes one chunk of the data at a time (for real-time streaming use cases). It applies EWMA to smooth
    the data and uses Z-score to detect anomalies.
    """
    ewma_values = []
    anomalies = []
    history = deque(maxlen=50)  # Rolling window of past data points for computing EWMA, mean, and std dev

    # Iterate over each data point in the chunk
    for value in data_chunk:
        # Calculate average and standard deviation based on history
        if len(history) > 1:
            avg, std_dev = np.mean(history), np.std(history)
        else:
            avg, std_dev = value, 1  # Initialize avg and std_dev during the first iteration
        
        # Dynamically adjust alpha if adaptive_alpha is enabled
        if adaptive_alpha:
            volatility = np.std(history) if len(history) > 1 else 0
            alpha = min(0.5, max(0.1, volatility / 10))  # High volatility -> faster adaptation
        else:
            alpha = 0.2  # Default alpha (slow adaptation to new data)

        ewma, z_score = ewma_z_score(value, history, avg, std_dev, alpha)
        ewma_values.append(ewma)
        
        # Detect anomalies where the Z-score is above the threshold (3 standard deviations from the mean)
        if abs(z_score) > 3:
            anomalies.append((len(ewma_values) - 1, value))

        # Add the current value to history for future calculations
        history.append(value)

    return ewma_values, anomalies

# Step 4: Visualize the data, EWMA, and detected anomalies
def visualize_stream(data_chunk, ewma_values=None, anomalies=None):
    """
    Generate a plot showing the original data, the EWMA smoothed data, and any detected anomalies.

    Parameters:
    - data_chunk (array): The data to visualize.
    - ewma_values (list): The calculated EWMA values for the data.
    - anomalies (list of tuples): The detected anomalies (index, value) pairs.
    
    This function uses matplotlib to plot the data, and highlights the detected anomalies (outliers) in red.
    """
    plt.figure(figsize=(12, 6))

    # Plot the original data
    plt.plot(data_chunk, label="Data Stream", color="blue")

    # Plot the EWMA (smoothed data) if available
    if ewma_values is not None:
        plt.plot(ewma_values, label="EWMA", color="green", linestyle="--")

    # Highlight anomalies as red dots
    if anomalies:
        anomaly_indices, anomaly_values = zip(*anomalies) if anomalies else ([], [])
        plt.scatter(anomaly_indices, anomaly_values, color="red", label="Anomalies", zorder=5)

    plt.title("Data Stream with Detected Anomalies")
    plt.xlabel("Time (Ticks)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# Step 5: Simulate continuous data stream chunks
def continuous_data_stream(size=300, chunk_size=50):
    """
    Simulate continuous real-time data streaming by yielding data in chunks.

    Parameters:
    - size (int): The total number of data points to generate.
    - chunk_size (int): The number of data points to process at once (mimicking real-time chunks).

    Yields:
    - Chunks of the data stream.
    """
    full_stream = generate_dynamic_data_stream(size)
    for i in range(0, size, chunk_size):
        yield full_stream[i:i + chunk_size]

# Step 6: Process each data chunk in real time, detecting and displaying anomalies
def process_data_in_real_time():
    """
    Continuously process and visualize the data stream, chunk by chunk, detecting anomalies in real-time.

    This function is designed to simulate real-time data processing, where chunks of data are processed sequentially.
    After each chunk is processed, the detected anomalies and the EWMA smoothed data are visualized.
    """
    stream = continuous_data_stream()

    # Initialize a counter for the total number of detected anomalies
    total_anomalies = 0

    # Process each chunk of data in sequence
    for chunk_number, data_chunk in enumerate(stream, start=1):
        print(f"Processing chunk {chunk_number}...\n")
        ewma_values, anomalies = detect_anomalies(data_chunk)
        total_anomalies += len(anomalies)

        print(f"Chunk {chunk_number} processed: {len(anomalies)} anomalies detected.")
        print(f"Total anomalies detected so far: {total_anomalies}\n")

        visualize_stream(data_chunk, ewma_values, anomalies)

        # Pause to allow user to review the current chunk's results before continuing to the next one
        input("Press Enter to continue to the next chunk...")

# Step 7: Validate the data to ensure no NaN or infinite values exist
def validate_data(data_stream):
    """
    Validate the data stream to ensure it contains no invalid values (NaNs or infinite values).

    Parameters:
    - data_stream: The NumPy array of data to validate.

    Raises:
    - ValueError: If any NaN or infinite values are found in the data.
    
    This function ensures the data stream is valid and ready for processing, avoiding errors during anomaly detection.
    """
    if not isinstance(data_stream, np.ndarray):
        raise ValueError("Data stream must be a numpy array.")
    if np.isnan(data_stream).any() or np.isinf(data_stream).any():
        raise ValueError("Data stream contains NaN or infinite values.")

# Main script execution
if __name__ == "__main__":
    try:
        # Process and visualize the data stream in real-time
        process_data_in_real_time()
    except Exception as e:
        print(f"Error occurred: {e}")
