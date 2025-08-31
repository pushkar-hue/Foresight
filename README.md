# Foresight: Short-Term Energy Load Forecasting 

> A high-performance, Bidirectional LSTM-based model for short-term energy load forecasting. Foresight is designed to help decarbonize buildings by enabling intelligent energy management and reducing waste.

## Inspiration

Buildings are one of the largest silent contributors to the climate crisis, consuming about a third of the world's energy. The inspiration for this project came from a simple idea: **decarbonization isn't just about switching to renewable energy, it's about using that energy intelligently.**

Accurate short-term load forecasting is the key. By predicting a building's energy needs in the near future, we can optimize heating, cooling, and lighting, better integrate unpredictable renewables like solar, and reduce our reliance on fossil fuel power plants. This project aims to create a scalable tool that brings this predictive power to any building, accelerating our transition to a sustainable future.

-----

## What It Does

Foresight leverages a **Bidirectional LSTM (BiLSTM) model** to produce highly accurate short-term forecasts of energy consumption. The model achieves an exceptionally low prediction loss, providing a reliable and precise view of future demand.

These forecasts can be used for:

  * **Smart Energy Management:** Optimize energy usage in real-time.
  * **Anomaly Detection:** Identify faulty components or unusual consumption patterns.
  * **Grid Integration:** Improve the integration of renewable energy sources.

-----

## Model Architecture & Performance

The core of our model is a Bidirectional Long Short-Term Memory (BiLSTM) network. While a standard LSTM processes data chronologically (past to future), a BiLSTM processes data in **two directions: forwards and backwards**. This allows the model to learn from the full context of a data point, considering both what happened before and what came after.

The final hidden state $h\_t$ at any given time step $t$ is a concatenation of the forward hidden state $\\vec{h\_t}$ and the backward hidden state $\\overleftarrow{h\_t}$:

$$h_t = [\vec{h_t} ; \overleftarrow{h_t}]$$

This dual-context approach makes the model incredibly powerful for identifying subtle patterns in time-series data.

### Performance

We are proud to have achieved a **test RMSE of \~0.05** on the UCI "Individual household electric power consumption" dataset. This result significantly outperforms the previous top score of \~0.6 on the same dataset.

The model demonstrates excellent learning behavior with both training and validation losses converging to near-zero, indicating a robust and well-generalized model.
![WhatsApp Image 2025-08-30 at 21 06 37_1ac57376](https://github.com/user-attachments/assets/7ef54fdb-4e12-4eee-8480-36ccf85c9ce3)
![WhatsApp Image 2025-08-30 at 21 06 43_bc06611c](https://github.com/user-attachments/assets/0ef83f28-4364-480d-82d1-c287195acae7)

-----

## 🛠Getting Started

### Prerequisites

  * Python 3.8+
  * TensorFlow / Keras
  * Pandas
  * NumPy
  * Scikit-learn

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/pushkar-hue/foresight.git
    cd foresight
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Challenges & Future Work

### Challenges

  * **Compute Limitations:** As students, we relied on the free tiers of Google Colab and Kaggle, which limited the scale and speed of our experimentation.
  * **Overfitting:** We initially faced issues with high variance where the model was overfitting the training data. This was addressed through careful regularization and model tuning.

### What's Next?

We believe we can push the performance even further. Our immediate focus is on exploring **attention-based mechanisms** and Transformer architectures, which we hypothesize will improve the model's ability to capture long-range dependencies in the data.

-----

## Cotribution:
* [Ved Thorat](https://github.com/i3hz/)
* [Pushkar Sharma](https://github.com/pushkar-hue/)
* [Shiva]
