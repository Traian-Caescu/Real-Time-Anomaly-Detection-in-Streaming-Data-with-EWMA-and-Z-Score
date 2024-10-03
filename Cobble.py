import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

# Step 1: Simulate a dynamic data stream
def generate_dynamic_data_stream(size, anomaly_prob=0.01, anomaly_magnitude=10, noise_scale=0.5): 
    """
    Simulate a continuous data stream, incorporating seasonal variations, random noise,
    and occasional anomalies. The goal is to create a stream that mimics real-world
    scenarios such as financial data or sensor readings.

    Parameters:
    - size (int): Total number of data points in the stream.
    - anomaly_prob (float): Probability of anomalies occurring in the stream.
    - anomaly_magnitude (float): Size or strength of the anomalies.
    - noise_scale (float): Degree of random noise added to simulate real-world imperfections.

    Returns:
    - data (array): Simulated stream of data points with anomalies.

    This function generates data with both predictable (seasonal and trend-based) 
    and unpredictable (random noise and anomalies) components.
    
    Note: Seasonal components help mimic periodic fluctuations while anomalies 
    represent unusual deviations. This data can simulate various domains like 
    financial systems or sensor data streams.
    """
    time = np.arange(size)  # Generate time points for the data stream
    
    # Seasonal component to simulate periodic fluctuations in the data
    seasonal_component = 5 * np.sin(2 * np.pi * time / 50) + 2 * np.sin(2 * np.pi * time / 150)
    
    # Random noise to make the data less deterministic and more realistic
    noise = np.random.normal(0, noise_scale, size)
    
    # Add a trend to simulate a gradual increase or decrease over time
    trend = 0.01 * time
    
    # Combine base value (10) with seasonal patterns, noise, and trend
    data = 10 + seasonal_component + noise + trend

    # Inject anomalies (e.g., sudden spikes, dips, or shifts in the data)
    anomalies_indices = np.random.choice(size, int(size * anomaly_prob), replace=False)
    anomaly_types = np.random.choice(['spike', 'dip', 'shift'], size=len(anomalies_indices))

    # Apply the anomalies in the data stream
    for i, anomaly_type in zip(anomalies_indices, anomaly_types):
        try:
            if anomaly_type == 'spike':
                data[i] += anomaly_magnitude  # A sudden upward spike
            elif anomaly_type == 'dip':
                data[i] -= anomaly_magnitude  # A sharp drop
            elif anomaly_type == 'shift':
                # A gradual shift over several data points
                end_idx = min(i + 5, size)
                data[i:end_idx] += anomaly_magnitude / 2 if random.random() > 0.5 else -anomaly_magnitude / 2
        except Exception as e:
            print(f"Error injecting anomaly: {e}")

    return data

# Step 2: Exponentially Weighted Moving Average (EWMA) and Z-score calculation
def ewma_z_score(new_value, history, avg, std_dev, alpha=0.2):
    """
    Calculate the EWMA for smoothing the data and the Z-score for anomaly detection.

    Parameters:
    - new_value: Latest data point in the stream.
    - history (deque): Buffer of recent values to compute statistics (e.g., mean, std_dev).
    - avg (float): Current average of values in history.
    - std_dev (float): Current standard deviation of values in history.
    - alpha (float): Smoothing factor for EWMA. Controls how fast it adapts to new values.

    Returns:
    - ewma (float): Smoothed EWMA value for the current data point.
    - z_score (float): Z-score indicating deviation of the current data point from the mean.

    EWMA smooths out short-term fluctuations and highlights long-term trends. Z-score 
    flags outliers by measuring how far a point deviates from the historical average.

    Why EWMA? 
    EWMA is particularly useful for real-time anomaly detection because it provides 
    a smooth and adaptive mechanism to capture trends while responding to recent data points. 
    Its adaptability to changes over time (like concept drift) makes it ideal for dynamic data.
    """
    try:
        ewma = (1 - alpha) * history[-1] + alpha * new_value if history else new_value
        z_score = (new_value - avg) / std_dev if std_dev != 0 else 0
    except Exception as e:
        print(f"Error calculating EWMA or Z-score: {e}")
        ewma, z_score = new_value, 0  # Fallback in case of error
    return ewma, z_score

# Step 3: Detect anomalies in a chunk of data using EWMA and Z-score
def detect_anomalies(data_chunk, adaptive_alpha=False, z_threshold=3):
    """
    Detect anomalies in a chunk of data based on the EWMA and Z-score methods.

    Parameters:
    - data_chunk (list/array): Segment of data to be analyzed.
    - adaptive_alpha (bool): Whether to adjust the EWMA smoothing factor based on data volatility.
    - z_threshold (float): Threshold for Z-score to flag anomalies. Default is 3.

    Returns:
    - ewma_values (list): List of EWMA values for each data point.
    - anomalies (list): List of detected anomalies as (index, value) pairs.

    This function processes one chunk of data at a time (mimicking real-time streaming)
    and identifies outliers based on how much they deviate from the recent trend.
    
    Why Z-score? 
    Z-score flags points that are more than 3 standard deviations away from the mean, 
    making it a highly effective method for detecting both gradual shifts and sudden spikes.
    """
    ewma_values = []  # Store EWMA values
    anomalies = []    # Store detected anomalies
    history = deque(maxlen=50)  # Rolling window of past values

    for value in data_chunk:
        try:
            if len(history) > 1:
                avg, std_dev = np.mean(history), np.std(history)  # Compute average and standard deviation
            else:
                avg, std_dev = value, 1  # Initialize in the first iteration

            # Adjust alpha based on volatility, if adaptive_alpha is enabled
            if adaptive_alpha:
                volatility = np.std(history) if len(history) > 1 else 0
                alpha = min(0.5, max(0.1, volatility / 10))  # Higher volatility -> faster adaptation
            else:
                alpha = 0.2  # Default alpha (slow adaptation to new data)

            ewma, z_score = ewma_z_score(value, history, avg, std_dev, alpha)
            ewma_values.append(ewma)

            # Flag anomalies if the Z-score exceeds the threshold (default 3)
            if abs(z_score) > z_threshold:
                anomalies.append((len(ewma_values) - 1, value))

            history.append(value)
        except Exception as e:
            print(f"Error detecting anomaly: {e}")

    return ewma_values, anomalies

# Step 4: Real-time visualization using Matplotlib animation
def real_time_visualization(stream, ewma_values=None, anomalies=None):
    """
    Visualize the data stream in real-time along with detected anomalies and the EWMA trend.

    Parameters:
    - stream (list/array): The full data stream.
    - ewma_values (list): Smoothed EWMA values.
    - anomalies (list): List of detected anomalies as (index, value) pairs.

    This function uses Matplotlib's animation capabilities to simulate real-time 
    visualization, where the data stream and anomalies are plotted incrementally.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('Real-time Data Stream with Anomalies')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.grid(True)

    line_data, = ax.plot([], [], lw=2, label="Data Stream")
    line_ewma, = ax.plot([], [], lw=2, linestyle="--", label="EWMA", color="green")
    anomaly_scatter = ax.scatter([], [], color="red", zorder=5, label="Anomalies")

    ax.legend()

    def update(i):
        try:
            ax.set_xlim(0, len(stream))  # Set the x-axis range based on the data size
            ax.set_ylim(min(stream) - 5, max(stream) + 5)  # Adjust y-axis range dynamically

            x_data = np.arange(len(stream[:i]))
            line_data.set_data(x_data, stream[:i])

            if ewma_values:
                line_ewma.set_data(x_data, ewma_values[:i])

            if anomalies:
                anomaly_indices, anomaly_values = zip(*anomalies) if anomalies else ([], [])
                anomaly_scatter.set_offsets(np.c_[anomaly_indices[:i], anomaly_values[:i]])

        except Exception as e:
            print(f"Error updating real-time visualization: {e}")

        return line_data, line_ewma, anomaly_scatter

    ani = animation.FuncAnimation(fig, update, frames=np.arange(len(stream)), interval=200)
    plt.show()

# Step 5: Simulate continuous real-time data stream in chunks
def continuous_data_stream(size=300, chunk_size=50):
    """
    Generate a continuous data stream, yielding chunks of data for real-time processing.

    Parameters:
    - size (int): Total size of the data stream.
    - chunk_size (int): Number of data points to process in each chunk.

    Yields:
    - Chunks of data, simulating real-time streaming.
    """
    try:
        full_stream = generate_dynamic_data_stream(size)
        for i in range(0, size, chunk_size):
            yield full_stream[i:i + chunk_size]
    except Exception as e:
        print(f"Error generating data stream: {e}")

# Step 6: Process data stream chunk-by-chunk in real-time
def process_data_in_real_time():
    """
    Main function to process the simulated data stream in real-time. It handles
    detection of anomalies and visualizes each chunk of the data as it arrives.
    """
    stream = continuous_data_stream()  # Simulate the continuous data stream
    total_anomalies = 0  # Track the total number of anomalies detected

    for chunk_number, data_chunk in enumerate(stream, start=1):
        try:
            ewma_values, anomalies = detect_anomalies(data_chunk)  # Detect anomalies in the current chunk
            total_anomalies += len(anomalies)
            print(f"Chunk {chunk_number}: {len(anomalies)} anomalies detected.")  # Log anomaly count for each chunk
            real_time_visualization(data_chunk, ewma_values, anomalies)  # Visualize the results in real-time
        except Exception as e:
            print(f"Error processing data chunk {chunk_number}: {e}")

if __name__ == "__main__":
    process_data_in_real_time()  # Start processing the data stream in real-time
