# Neuralmedic - Kortex Gemini

This Django application provides real-time visualization of EEG data across 8 channels. The data is simulated using random values and displayed using Chart.js with the `chartjs-plugin-streaming` plugin for real-time updates.

## Features

- Real-time graph of 8 EEG channels combined.
- Individual real-time graphs for each EEG channel.
- WebSocket-based communication for near-zero latency data updates.

## kortex
Streamlit chatbot for EEG applications

### Steps

1. **Clone the repository:**
    ```bash
    git clone https://github.com/NeuralMedic-DE/kortex
    cd EEG-RTV
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
	
4. **Run streamlit app:**
    ```bash
    streamlit run kortex.py
    ```

## Requirements

- Python 3.7+
- Django 3.2+
- Channels 3.0+
- Redis Server

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/NeuralMedic-DE/EEG-RTV.git
    cd EEG-RTV
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Start Redis server:**
    - For Ubuntu/Debian:
        ```bash
        sudo apt update
        sudo apt install redis-server
        sudo systemctl start redis
        sudo systemctl enable redis
        ```
    - For macOS using Homebrew:
        ```bash
        brew install redis
        brew services start redis
        ```

5. **Apply migrations:**
    ```bash
   cd eeg_project
    python manage.py migrate
    ```

6. **Run the Django development server:**
    ```bash
    python manage.py runserver
    ```

7. **Access the application:**
    Open your browser and navigate to `http://127.0.0.1:8000/eeg_app/` to see the real-time EEG data visualization.

## Project Structure

- `eeg_project/`: The main Django project directory.
  - `asgi.py`: ASGI configuration for Channels.
  - `settings.py`: Project settings.
  - `urls.py`: URL configuration.
- `eeg_app/`: The Django app for EEG visualization.
  - `consumers.py`: WebSocket consumer to simulate and send EEG data.
  - `routing.py`: WebSocket routing configuration.
  - `views.py`: View to render the HTML template.
  - `templates/eeg_app/index.html`: HTML template containing the Chart.js setup.

## Notes

- Ensure Redis is running before starting the Django server.
- Adjust the frequency and data range in `EEGConsumer` as needed for your specific use case.


