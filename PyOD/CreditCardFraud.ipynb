{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "CreditCardFraud.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "kwektjq9Mhxr",
        "colab_type": "code",
        "outputId": "0a741a23-bf79-47e3-bb99-10b88ff44ca0",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 128
        }
      },
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly\n",
            "\n",
            "Enter your authorization code:\n",
            "··········\n",
            "Mounted at /content/drive\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "j4G9L6F2MsnV",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from matplotlib import pyplot as plt\n",
        "import warnings\n",
        "warnings.filterwarnings(\"ignore\")\n",
        "import seaborn as sns\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from imblearn.over_sampling import SMOTE\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.model_selection import train_test_split\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.layers import Dense\n",
        "from tensorflow.keras.layers import Input\n",
        "from tensorflow.keras.models import Model"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "gVo9a4-pNLBz",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "df = pd.read_csv(\"/content/drive/My Drive/creditcard.csv\", encoding=\"utf-8\")"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_jHkCgNkNT6L",
        "colab_type": "code",
        "outputId": "7fbc9944-09e7-4644-946d-f23b5548351a",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 427
        }
      },
      "source": [
        "df"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Time</th>\n",
              "      <th>V1</th>\n",
              "      <th>V2</th>\n",
              "      <th>V3</th>\n",
              "      <th>V4</th>\n",
              "      <th>V5</th>\n",
              "      <th>V6</th>\n",
              "      <th>V7</th>\n",
              "      <th>V8</th>\n",
              "      <th>V9</th>\n",
              "      <th>V10</th>\n",
              "      <th>V11</th>\n",
              "      <th>V12</th>\n",
              "      <th>V13</th>\n",
              "      <th>V14</th>\n",
              "      <th>V15</th>\n",
              "      <th>V16</th>\n",
              "      <th>V17</th>\n",
              "      <th>V18</th>\n",
              "      <th>V19</th>\n",
              "      <th>V20</th>\n",
              "      <th>V21</th>\n",
              "      <th>V22</th>\n",
              "      <th>V23</th>\n",
              "      <th>V24</th>\n",
              "      <th>V25</th>\n",
              "      <th>V26</th>\n",
              "      <th>V27</th>\n",
              "      <th>V28</th>\n",
              "      <th>Amount</th>\n",
              "      <th>Class</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.0</td>\n",
              "      <td>-1.359807</td>\n",
              "      <td>-0.072781</td>\n",
              "      <td>2.536347</td>\n",
              "      <td>1.378155</td>\n",
              "      <td>-0.338321</td>\n",
              "      <td>0.462388</td>\n",
              "      <td>0.239599</td>\n",
              "      <td>0.098698</td>\n",
              "      <td>0.363787</td>\n",
              "      <td>0.090794</td>\n",
              "      <td>-0.551600</td>\n",
              "      <td>-0.617801</td>\n",
              "      <td>-0.991390</td>\n",
              "      <td>-0.311169</td>\n",
              "      <td>1.468177</td>\n",
              "      <td>-0.470401</td>\n",
              "      <td>0.207971</td>\n",
              "      <td>0.025791</td>\n",
              "      <td>0.403993</td>\n",
              "      <td>0.251412</td>\n",
              "      <td>-0.018307</td>\n",
              "      <td>0.277838</td>\n",
              "      <td>-0.110474</td>\n",
              "      <td>0.066928</td>\n",
              "      <td>0.128539</td>\n",
              "      <td>-0.189115</td>\n",
              "      <td>0.133558</td>\n",
              "      <td>-0.021053</td>\n",
              "      <td>149.62</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.0</td>\n",
              "      <td>1.191857</td>\n",
              "      <td>0.266151</td>\n",
              "      <td>0.166480</td>\n",
              "      <td>0.448154</td>\n",
              "      <td>0.060018</td>\n",
              "      <td>-0.082361</td>\n",
              "      <td>-0.078803</td>\n",
              "      <td>0.085102</td>\n",
              "      <td>-0.255425</td>\n",
              "      <td>-0.166974</td>\n",
              "      <td>1.612727</td>\n",
              "      <td>1.065235</td>\n",
              "      <td>0.489095</td>\n",
              "      <td>-0.143772</td>\n",
              "      <td>0.635558</td>\n",
              "      <td>0.463917</td>\n",
              "      <td>-0.114805</td>\n",
              "      <td>-0.183361</td>\n",
              "      <td>-0.145783</td>\n",
              "      <td>-0.069083</td>\n",
              "      <td>-0.225775</td>\n",
              "      <td>-0.638672</td>\n",
              "      <td>0.101288</td>\n",
              "      <td>-0.339846</td>\n",
              "      <td>0.167170</td>\n",
              "      <td>0.125895</td>\n",
              "      <td>-0.008983</td>\n",
              "      <td>0.014724</td>\n",
              "      <td>2.69</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1.0</td>\n",
              "      <td>-1.358354</td>\n",
              "      <td>-1.340163</td>\n",
              "      <td>1.773209</td>\n",
              "      <td>0.379780</td>\n",
              "      <td>-0.503198</td>\n",
              "      <td>1.800499</td>\n",
              "      <td>0.791461</td>\n",
              "      <td>0.247676</td>\n",
              "      <td>-1.514654</td>\n",
              "      <td>0.207643</td>\n",
              "      <td>0.624501</td>\n",
              "      <td>0.066084</td>\n",
              "      <td>0.717293</td>\n",
              "      <td>-0.165946</td>\n",
              "      <td>2.345865</td>\n",
              "      <td>-2.890083</td>\n",
              "      <td>1.109969</td>\n",
              "      <td>-0.121359</td>\n",
              "      <td>-2.261857</td>\n",
              "      <td>0.524980</td>\n",
              "      <td>0.247998</td>\n",
              "      <td>0.771679</td>\n",
              "      <td>0.909412</td>\n",
              "      <td>-0.689281</td>\n",
              "      <td>-0.327642</td>\n",
              "      <td>-0.139097</td>\n",
              "      <td>-0.055353</td>\n",
              "      <td>-0.059752</td>\n",
              "      <td>378.66</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1.0</td>\n",
              "      <td>-0.966272</td>\n",
              "      <td>-0.185226</td>\n",
              "      <td>1.792993</td>\n",
              "      <td>-0.863291</td>\n",
              "      <td>-0.010309</td>\n",
              "      <td>1.247203</td>\n",
              "      <td>0.237609</td>\n",
              "      <td>0.377436</td>\n",
              "      <td>-1.387024</td>\n",
              "      <td>-0.054952</td>\n",
              "      <td>-0.226487</td>\n",
              "      <td>0.178228</td>\n",
              "      <td>0.507757</td>\n",
              "      <td>-0.287924</td>\n",
              "      <td>-0.631418</td>\n",
              "      <td>-1.059647</td>\n",
              "      <td>-0.684093</td>\n",
              "      <td>1.965775</td>\n",
              "      <td>-1.232622</td>\n",
              "      <td>-0.208038</td>\n",
              "      <td>-0.108300</td>\n",
              "      <td>0.005274</td>\n",
              "      <td>-0.190321</td>\n",
              "      <td>-1.175575</td>\n",
              "      <td>0.647376</td>\n",
              "      <td>-0.221929</td>\n",
              "      <td>0.062723</td>\n",
              "      <td>0.061458</td>\n",
              "      <td>123.50</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>2.0</td>\n",
              "      <td>-1.158233</td>\n",
              "      <td>0.877737</td>\n",
              "      <td>1.548718</td>\n",
              "      <td>0.403034</td>\n",
              "      <td>-0.407193</td>\n",
              "      <td>0.095921</td>\n",
              "      <td>0.592941</td>\n",
              "      <td>-0.270533</td>\n",
              "      <td>0.817739</td>\n",
              "      <td>0.753074</td>\n",
              "      <td>-0.822843</td>\n",
              "      <td>0.538196</td>\n",
              "      <td>1.345852</td>\n",
              "      <td>-1.119670</td>\n",
              "      <td>0.175121</td>\n",
              "      <td>-0.451449</td>\n",
              "      <td>-0.237033</td>\n",
              "      <td>-0.038195</td>\n",
              "      <td>0.803487</td>\n",
              "      <td>0.408542</td>\n",
              "      <td>-0.009431</td>\n",
              "      <td>0.798278</td>\n",
              "      <td>-0.137458</td>\n",
              "      <td>0.141267</td>\n",
              "      <td>-0.206010</td>\n",
              "      <td>0.502292</td>\n",
              "      <td>0.219422</td>\n",
              "      <td>0.215153</td>\n",
              "      <td>69.99</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>284802</th>\n",
              "      <td>172786.0</td>\n",
              "      <td>-11.881118</td>\n",
              "      <td>10.071785</td>\n",
              "      <td>-9.834783</td>\n",
              "      <td>-2.066656</td>\n",
              "      <td>-5.364473</td>\n",
              "      <td>-2.606837</td>\n",
              "      <td>-4.918215</td>\n",
              "      <td>7.305334</td>\n",
              "      <td>1.914428</td>\n",
              "      <td>4.356170</td>\n",
              "      <td>-1.593105</td>\n",
              "      <td>2.711941</td>\n",
              "      <td>-0.689256</td>\n",
              "      <td>4.626942</td>\n",
              "      <td>-0.924459</td>\n",
              "      <td>1.107641</td>\n",
              "      <td>1.991691</td>\n",
              "      <td>0.510632</td>\n",
              "      <td>-0.682920</td>\n",
              "      <td>1.475829</td>\n",
              "      <td>0.213454</td>\n",
              "      <td>0.111864</td>\n",
              "      <td>1.014480</td>\n",
              "      <td>-0.509348</td>\n",
              "      <td>1.436807</td>\n",
              "      <td>0.250034</td>\n",
              "      <td>0.943651</td>\n",
              "      <td>0.823731</td>\n",
              "      <td>0.77</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>284803</th>\n",
              "      <td>172787.0</td>\n",
              "      <td>-0.732789</td>\n",
              "      <td>-0.055080</td>\n",
              "      <td>2.035030</td>\n",
              "      <td>-0.738589</td>\n",
              "      <td>0.868229</td>\n",
              "      <td>1.058415</td>\n",
              "      <td>0.024330</td>\n",
              "      <td>0.294869</td>\n",
              "      <td>0.584800</td>\n",
              "      <td>-0.975926</td>\n",
              "      <td>-0.150189</td>\n",
              "      <td>0.915802</td>\n",
              "      <td>1.214756</td>\n",
              "      <td>-0.675143</td>\n",
              "      <td>1.164931</td>\n",
              "      <td>-0.711757</td>\n",
              "      <td>-0.025693</td>\n",
              "      <td>-1.221179</td>\n",
              "      <td>-1.545556</td>\n",
              "      <td>0.059616</td>\n",
              "      <td>0.214205</td>\n",
              "      <td>0.924384</td>\n",
              "      <td>0.012463</td>\n",
              "      <td>-1.016226</td>\n",
              "      <td>-0.606624</td>\n",
              "      <td>-0.395255</td>\n",
              "      <td>0.068472</td>\n",
              "      <td>-0.053527</td>\n",
              "      <td>24.79</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>284804</th>\n",
              "      <td>172788.0</td>\n",
              "      <td>1.919565</td>\n",
              "      <td>-0.301254</td>\n",
              "      <td>-3.249640</td>\n",
              "      <td>-0.557828</td>\n",
              "      <td>2.630515</td>\n",
              "      <td>3.031260</td>\n",
              "      <td>-0.296827</td>\n",
              "      <td>0.708417</td>\n",
              "      <td>0.432454</td>\n",
              "      <td>-0.484782</td>\n",
              "      <td>0.411614</td>\n",
              "      <td>0.063119</td>\n",
              "      <td>-0.183699</td>\n",
              "      <td>-0.510602</td>\n",
              "      <td>1.329284</td>\n",
              "      <td>0.140716</td>\n",
              "      <td>0.313502</td>\n",
              "      <td>0.395652</td>\n",
              "      <td>-0.577252</td>\n",
              "      <td>0.001396</td>\n",
              "      <td>0.232045</td>\n",
              "      <td>0.578229</td>\n",
              "      <td>-0.037501</td>\n",
              "      <td>0.640134</td>\n",
              "      <td>0.265745</td>\n",
              "      <td>-0.087371</td>\n",
              "      <td>0.004455</td>\n",
              "      <td>-0.026561</td>\n",
              "      <td>67.88</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>284805</th>\n",
              "      <td>172788.0</td>\n",
              "      <td>-0.240440</td>\n",
              "      <td>0.530483</td>\n",
              "      <td>0.702510</td>\n",
              "      <td>0.689799</td>\n",
              "      <td>-0.377961</td>\n",
              "      <td>0.623708</td>\n",
              "      <td>-0.686180</td>\n",
              "      <td>0.679145</td>\n",
              "      <td>0.392087</td>\n",
              "      <td>-0.399126</td>\n",
              "      <td>-1.933849</td>\n",
              "      <td>-0.962886</td>\n",
              "      <td>-1.042082</td>\n",
              "      <td>0.449624</td>\n",
              "      <td>1.962563</td>\n",
              "      <td>-0.608577</td>\n",
              "      <td>0.509928</td>\n",
              "      <td>1.113981</td>\n",
              "      <td>2.897849</td>\n",
              "      <td>0.127434</td>\n",
              "      <td>0.265245</td>\n",
              "      <td>0.800049</td>\n",
              "      <td>-0.163298</td>\n",
              "      <td>0.123205</td>\n",
              "      <td>-0.569159</td>\n",
              "      <td>0.546668</td>\n",
              "      <td>0.108821</td>\n",
              "      <td>0.104533</td>\n",
              "      <td>10.00</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>284806</th>\n",
              "      <td>172792.0</td>\n",
              "      <td>-0.533413</td>\n",
              "      <td>-0.189733</td>\n",
              "      <td>0.703337</td>\n",
              "      <td>-0.506271</td>\n",
              "      <td>-0.012546</td>\n",
              "      <td>-0.649617</td>\n",
              "      <td>1.577006</td>\n",
              "      <td>-0.414650</td>\n",
              "      <td>0.486180</td>\n",
              "      <td>-0.915427</td>\n",
              "      <td>-1.040458</td>\n",
              "      <td>-0.031513</td>\n",
              "      <td>-0.188093</td>\n",
              "      <td>-0.084316</td>\n",
              "      <td>0.041333</td>\n",
              "      <td>-0.302620</td>\n",
              "      <td>-0.660377</td>\n",
              "      <td>0.167430</td>\n",
              "      <td>-0.256117</td>\n",
              "      <td>0.382948</td>\n",
              "      <td>0.261057</td>\n",
              "      <td>0.643078</td>\n",
              "      <td>0.376777</td>\n",
              "      <td>0.008797</td>\n",
              "      <td>-0.473649</td>\n",
              "      <td>-0.818267</td>\n",
              "      <td>-0.002415</td>\n",
              "      <td>0.013649</td>\n",
              "      <td>217.00</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>284807 rows × 31 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "            Time         V1         V2  ...       V28  Amount  Class\n",
              "0            0.0  -1.359807  -0.072781  ... -0.021053  149.62      0\n",
              "1            0.0   1.191857   0.266151  ...  0.014724    2.69      0\n",
              "2            1.0  -1.358354  -1.340163  ... -0.059752  378.66      0\n",
              "3            1.0  -0.966272  -0.185226  ...  0.061458  123.50      0\n",
              "4            2.0  -1.158233   0.877737  ...  0.215153   69.99      0\n",
              "...          ...        ...        ...  ...       ...     ...    ...\n",
              "284802  172786.0 -11.881118  10.071785  ...  0.823731    0.77      0\n",
              "284803  172787.0  -0.732789  -0.055080  ... -0.053527   24.79      0\n",
              "284804  172788.0   1.919565  -0.301254  ... -0.026561   67.88      0\n",
              "284805  172788.0  -0.240440   0.530483  ...  0.104533   10.00      0\n",
              "284806  172792.0  -0.533413  -0.189733  ...  0.013649  217.00      0\n",
              "\n",
              "[284807 rows x 31 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0OSx8rjZONgT",
        "colab_type": "code",
        "outputId": "3e8ceb0b-dd0d-4373-fd5d-a5731b649606",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 218
        }
      },
      "source": [
        "df.head()"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Time</th>\n",
              "      <th>V1</th>\n",
              "      <th>V2</th>\n",
              "      <th>V3</th>\n",
              "      <th>V4</th>\n",
              "      <th>V5</th>\n",
              "      <th>V6</th>\n",
              "      <th>V7</th>\n",
              "      <th>V8</th>\n",
              "      <th>V9</th>\n",
              "      <th>V10</th>\n",
              "      <th>V11</th>\n",
              "      <th>V12</th>\n",
              "      <th>V13</th>\n",
              "      <th>V14</th>\n",
              "      <th>V15</th>\n",
              "      <th>V16</th>\n",
              "      <th>V17</th>\n",
              "      <th>V18</th>\n",
              "      <th>V19</th>\n",
              "      <th>V20</th>\n",
              "      <th>V21</th>\n",
              "      <th>V22</th>\n",
              "      <th>V23</th>\n",
              "      <th>V24</th>\n",
              "      <th>V25</th>\n",
              "      <th>V26</th>\n",
              "      <th>V27</th>\n",
              "      <th>V28</th>\n",
              "      <th>Amount</th>\n",
              "      <th>Class</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.0</td>\n",
              "      <td>-1.359807</td>\n",
              "      <td>-0.072781</td>\n",
              "      <td>2.536347</td>\n",
              "      <td>1.378155</td>\n",
              "      <td>-0.338321</td>\n",
              "      <td>0.462388</td>\n",
              "      <td>0.239599</td>\n",
              "      <td>0.098698</td>\n",
              "      <td>0.363787</td>\n",
              "      <td>0.090794</td>\n",
              "      <td>-0.551600</td>\n",
              "      <td>-0.617801</td>\n",
              "      <td>-0.991390</td>\n",
              "      <td>-0.311169</td>\n",
              "      <td>1.468177</td>\n",
              "      <td>-0.470401</td>\n",
              "      <td>0.207971</td>\n",
              "      <td>0.025791</td>\n",
              "      <td>0.403993</td>\n",
              "      <td>0.251412</td>\n",
              "      <td>-0.018307</td>\n",
              "      <td>0.277838</td>\n",
              "      <td>-0.110474</td>\n",
              "      <td>0.066928</td>\n",
              "      <td>0.128539</td>\n",
              "      <td>-0.189115</td>\n",
              "      <td>0.133558</td>\n",
              "      <td>-0.021053</td>\n",
              "      <td>149.62</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.0</td>\n",
              "      <td>1.191857</td>\n",
              "      <td>0.266151</td>\n",
              "      <td>0.166480</td>\n",
              "      <td>0.448154</td>\n",
              "      <td>0.060018</td>\n",
              "      <td>-0.082361</td>\n",
              "      <td>-0.078803</td>\n",
              "      <td>0.085102</td>\n",
              "      <td>-0.255425</td>\n",
              "      <td>-0.166974</td>\n",
              "      <td>1.612727</td>\n",
              "      <td>1.065235</td>\n",
              "      <td>0.489095</td>\n",
              "      <td>-0.143772</td>\n",
              "      <td>0.635558</td>\n",
              "      <td>0.463917</td>\n",
              "      <td>-0.114805</td>\n",
              "      <td>-0.183361</td>\n",
              "      <td>-0.145783</td>\n",
              "      <td>-0.069083</td>\n",
              "      <td>-0.225775</td>\n",
              "      <td>-0.638672</td>\n",
              "      <td>0.101288</td>\n",
              "      <td>-0.339846</td>\n",
              "      <td>0.167170</td>\n",
              "      <td>0.125895</td>\n",
              "      <td>-0.008983</td>\n",
              "      <td>0.014724</td>\n",
              "      <td>2.69</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1.0</td>\n",
              "      <td>-1.358354</td>\n",
              "      <td>-1.340163</td>\n",
              "      <td>1.773209</td>\n",
              "      <td>0.379780</td>\n",
              "      <td>-0.503198</td>\n",
              "      <td>1.800499</td>\n",
              "      <td>0.791461</td>\n",
              "      <td>0.247676</td>\n",
              "      <td>-1.514654</td>\n",
              "      <td>0.207643</td>\n",
              "      <td>0.624501</td>\n",
              "      <td>0.066084</td>\n",
              "      <td>0.717293</td>\n",
              "      <td>-0.165946</td>\n",
              "      <td>2.345865</td>\n",
              "      <td>-2.890083</td>\n",
              "      <td>1.109969</td>\n",
              "      <td>-0.121359</td>\n",
              "      <td>-2.261857</td>\n",
              "      <td>0.524980</td>\n",
              "      <td>0.247998</td>\n",
              "      <td>0.771679</td>\n",
              "      <td>0.909412</td>\n",
              "      <td>-0.689281</td>\n",
              "      <td>-0.327642</td>\n",
              "      <td>-0.139097</td>\n",
              "      <td>-0.055353</td>\n",
              "      <td>-0.059752</td>\n",
              "      <td>378.66</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1.0</td>\n",
              "      <td>-0.966272</td>\n",
              "      <td>-0.185226</td>\n",
              "      <td>1.792993</td>\n",
              "      <td>-0.863291</td>\n",
              "      <td>-0.010309</td>\n",
              "      <td>1.247203</td>\n",
              "      <td>0.237609</td>\n",
              "      <td>0.377436</td>\n",
              "      <td>-1.387024</td>\n",
              "      <td>-0.054952</td>\n",
              "      <td>-0.226487</td>\n",
              "      <td>0.178228</td>\n",
              "      <td>0.507757</td>\n",
              "      <td>-0.287924</td>\n",
              "      <td>-0.631418</td>\n",
              "      <td>-1.059647</td>\n",
              "      <td>-0.684093</td>\n",
              "      <td>1.965775</td>\n",
              "      <td>-1.232622</td>\n",
              "      <td>-0.208038</td>\n",
              "      <td>-0.108300</td>\n",
              "      <td>0.005274</td>\n",
              "      <td>-0.190321</td>\n",
              "      <td>-1.175575</td>\n",
              "      <td>0.647376</td>\n",
              "      <td>-0.221929</td>\n",
              "      <td>0.062723</td>\n",
              "      <td>0.061458</td>\n",
              "      <td>123.50</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>2.0</td>\n",
              "      <td>-1.158233</td>\n",
              "      <td>0.877737</td>\n",
              "      <td>1.548718</td>\n",
              "      <td>0.403034</td>\n",
              "      <td>-0.407193</td>\n",
              "      <td>0.095921</td>\n",
              "      <td>0.592941</td>\n",
              "      <td>-0.270533</td>\n",
              "      <td>0.817739</td>\n",
              "      <td>0.753074</td>\n",
              "      <td>-0.822843</td>\n",
              "      <td>0.538196</td>\n",
              "      <td>1.345852</td>\n",
              "      <td>-1.119670</td>\n",
              "      <td>0.175121</td>\n",
              "      <td>-0.451449</td>\n",
              "      <td>-0.237033</td>\n",
              "      <td>-0.038195</td>\n",
              "      <td>0.803487</td>\n",
              "      <td>0.408542</td>\n",
              "      <td>-0.009431</td>\n",
              "      <td>0.798278</td>\n",
              "      <td>-0.137458</td>\n",
              "      <td>0.141267</td>\n",
              "      <td>-0.206010</td>\n",
              "      <td>0.502292</td>\n",
              "      <td>0.219422</td>\n",
              "      <td>0.215153</td>\n",
              "      <td>69.99</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   Time        V1        V2        V3  ...       V27       V28  Amount  Class\n",
              "0   0.0 -1.359807 -0.072781  2.536347  ...  0.133558 -0.021053  149.62      0\n",
              "1   0.0  1.191857  0.266151  0.166480  ... -0.008983  0.014724    2.69      0\n",
              "2   1.0 -1.358354 -1.340163  1.773209  ... -0.055353 -0.059752  378.66      0\n",
              "3   1.0 -0.966272 -0.185226  1.792993  ...  0.062723  0.061458  123.50      0\n",
              "4   2.0 -1.158233  0.877737  1.548718  ...  0.219422  0.215153   69.99      0\n",
              "\n",
              "[5 rows x 31 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4P77H59wO_7I",
        "colab_type": "code",
        "outputId": "6f6379d8-fa52-4877-8191-94fcb6761a90",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 638
        }
      },
      "source": [
        "fig, ax = plt.subplots(figsize=(20,10))\n",
        "corr = df.corr()\n",
        "sns.heatmap(corr, cmap=\"gray\", ax=ax)\n",
        "ax.set_title(\"Imbalanced Correlation Matrix\", fontsize=14)\n",
        "plt.show()"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1440x720 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "PD2LuGadG2SU",
        "colab_type": "code",
        "outputId": "dcdee32d-f879-4689-e130-9c5f5151dd3f",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 297
        }
      },
      "source": [
        "sns.countplot(x='Class',data=df)"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f690eb54f60>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 7
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tSmcDQp2PTYl",
        "colab_type": "code",
        "outputId": "5f00daa2-e2fe-4543-9290-9157b2e145b1",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 72
        }
      },
      "source": [
        "sm = SMOTE(sampling_strategy='minority', random_state=7)\n",
        "resampled_X, resampled_Y = sm.fit_resample(df.drop('Class', axis=1), df['Class'])\n",
        "oversampled_df = pd.concat([pd.DataFrame(resampled_X), pd.DataFrame(resampled_Y)], axis=1)\n",
        "oversampled_df.columns = df.columns\n",
        "oversampled_df['Class'].value_counts()"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1    284315\n",
              "0    284315\n",
              "Name: Class, dtype: int64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "CEmbC9BuHUaX",
        "colab_type": "code",
        "outputId": "5a7078be-cd0a-4338-8efc-4bf543b48f17",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 297
        }
      },
      "source": [
        "sns.countplot(x='Class', data=oversampled_df)"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f6908f63a20>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 10
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ynruVYzNPULL",
        "colab_type": "code",
        "outputId": "d317c03a-fb54-4e1b-a942-834aecc84f93",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 638
        }
      },
      "source": [
        "fig, ax = plt.subplots(figsize=(20,10))         \n",
        "corr = oversampled_df.corr()\n",
        "sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)\n",
        "ax.set_title(\"Imbalanced Correlation Matrix\", fontsize=14)\n",
        "plt.show()"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1440x720 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hpG33h50Pbci",
        "colab_type": "code",
        "outputId": "629c010f-fed0-4443-ca83-7ea1a1a4453f",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 126
        }
      },
      "source": [
        "sc = StandardScaler()\n",
        "X = oversampled_df.iloc[:, 1:-1].values\n",
        "y = oversampled_df.iloc[:, -1].values\n",
        "y = y.reshape(-1, 1)\n",
        "print(X.shape, y.shape)\n",
        "\n",
        "X = sc.fit_transform(X)\n",
        "print(X[0])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "(568630, 29) (568630, 1)\n",
            "[ 0.20495125 -0.54573636  1.0045184  -0.30168224  0.31098779  0.6920209\n",
            "  0.55516986 -0.03644768  0.76125221  0.67923829 -0.92061643  0.57128753\n",
            " -0.94634073  0.71704508  1.64724241  0.48636415  0.62706212  0.50679008\n",
            "  0.04977222  0.06379384 -0.14589103  0.24649907 -0.10441593  0.22564242\n",
            "  0.16596895 -0.48756152  0.05492719 -0.14984388  0.2455859 ]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vZo_zgJIPfAK",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xuUKCg7VPqt0",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "adam = tf.keras.optimizers.Adam(learning_rate=0.0005)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Y0AAGUpUQ_n9",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "x_features = X.shape[1]\n",
        "y_features = y.shape[1]"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "meQZvTPqRDCt",
        "colab_type": "code",
        "outputId": "81103b1e-8622-4b74-b127-379c83d70937",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 326
        }
      },
      "source": [
        "i = Input(shape=(x_features,))\n",
        "\n",
        "x = Dense(64, activation='relu')(i)\n",
        "x = Dense(64, activation='relu')(x)\n",
        "o = Dense(y_features, activation='sigmoid')(x)\n",
        "\n",
        "model = Model(i,o)\n",
        "model.compile(loss=\"binary_crossentropy\", metrics=['accuracy'], optimizer=adam)\n",
        "print(model.summary())\n",
        "callback = tf.keras.callbacks.EarlyStopping(\n",
        "    monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto',\n",
        "    baseline=None, restore_best_weights=True\n",
        ")"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Model: \"model\"\n",
            "_________________________________________________________________\n",
            "Layer (type)                 Output Shape              Param #   \n",
            "=================================================================\n",
            "input_1 (InputLayer)         [(None, 29)]              0         \n",
            "_________________________________________________________________\n",
            "dense (Dense)                (None, 64)                1920      \n",
            "_________________________________________________________________\n",
            "dense_1 (Dense)              (None, 64)                4160      \n",
            "_________________________________________________________________\n",
            "dense_2 (Dense)              (None, 1)                 65        \n",
            "=================================================================\n",
            "Total params: 6,145\n",
            "Trainable params: 6,145\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n",
            "None\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zH-_d5ZqRLrV",
        "colab_type": "code",
        "outputId": "972398f2-e27e-4be6-8ea3-4fa4c65e1472",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        }
      },
      "source": [
        "r = model.fit(x_train, y_train, epochs=100, batch_size=512, verbose=1, validation_data=(x_test, y_test), callbacks=[callback])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Epoch 1/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.1122 - accuracy: 0.9588 - val_loss: 0.0468 - val_accuracy: 0.9844\n",
            "Epoch 2/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0336 - accuracy: 0.9883 - val_loss: 0.0260 - val_accuracy: 0.9929\n",
            "Epoch 3/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0198 - accuracy: 0.9939 - val_loss: 0.0165 - val_accuracy: 0.9945\n",
            "Epoch 4/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0133 - accuracy: 0.9965 - val_loss: 0.0117 - val_accuracy: 0.9974\n",
            "Epoch 5/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0095 - accuracy: 0.9978 - val_loss: 0.0096 - val_accuracy: 0.9977\n",
            "Epoch 6/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0074 - accuracy: 0.9984 - val_loss: 0.0068 - val_accuracy: 0.9986\n",
            "Epoch 7/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0060 - accuracy: 0.9987 - val_loss: 0.0061 - val_accuracy: 0.9986\n",
            "Epoch 8/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0050 - accuracy: 0.9989 - val_loss: 0.0051 - val_accuracy: 0.9989\n",
            "Epoch 9/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0042 - accuracy: 0.9991 - val_loss: 0.0046 - val_accuracy: 0.9989\n",
            "Epoch 10/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0037 - accuracy: 0.9992 - val_loss: 0.0041 - val_accuracy: 0.9990\n",
            "Epoch 11/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0033 - accuracy: 0.9993 - val_loss: 0.0037 - val_accuracy: 0.9993\n",
            "Epoch 12/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0029 - accuracy: 0.9994 - val_loss: 0.0035 - val_accuracy: 0.9992\n",
            "Epoch 13/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0027 - accuracy: 0.9994 - val_loss: 0.0037 - val_accuracy: 0.9991\n",
            "Epoch 14/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0024 - accuracy: 0.9995 - val_loss: 0.0035 - val_accuracy: 0.9990\n",
            "Epoch 15/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0022 - accuracy: 0.9995 - val_loss: 0.0032 - val_accuracy: 0.9992\n",
            "Epoch 16/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0020 - accuracy: 0.9996 - val_loss: 0.0027 - val_accuracy: 0.9994\n",
            "Epoch 17/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0020 - accuracy: 0.9995 - val_loss: 0.0027 - val_accuracy: 0.9993\n",
            "Epoch 18/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0018 - accuracy: 0.9996 - val_loss: 0.0023 - val_accuracy: 0.9996\n",
            "Epoch 19/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0016 - accuracy: 0.9996 - val_loss: 0.0025 - val_accuracy: 0.9994\n",
            "Epoch 20/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0015 - accuracy: 0.9997 - val_loss: 0.0025 - val_accuracy: 0.9993\n",
            "Epoch 21/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0015 - accuracy: 0.9996 - val_loss: 0.0022 - val_accuracy: 0.9994\n",
            "Epoch 22/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0014 - accuracy: 0.9997 - val_loss: 0.0024 - val_accuracy: 0.9994\n",
            "Epoch 23/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0013 - accuracy: 0.9997 - val_loss: 0.0022 - val_accuracy: 0.9996\n",
            "Epoch 24/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0012 - accuracy: 0.9997 - val_loss: 0.0022 - val_accuracy: 0.9995\n",
            "Epoch 25/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0014 - accuracy: 0.9997 - val_loss: 0.0025 - val_accuracy: 0.9993\n",
            "Epoch 26/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0011 - accuracy: 0.9997 - val_loss: 0.0027 - val_accuracy: 0.9993\n",
            "Epoch 27/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 9.9902e-04 - accuracy: 0.9997 - val_loss: 0.0023 - val_accuracy: 0.9995\n",
            "Epoch 28/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 9.9616e-04 - accuracy: 0.9997 - val_loss: 0.0021 - val_accuracy: 0.9995\n",
            "Epoch 29/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 8.8531e-04 - accuracy: 0.9998 - val_loss: 0.0020 - val_accuracy: 0.9995\n",
            "Epoch 30/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 0.0010 - accuracy: 0.9997 - val_loss: 0.0018 - val_accuracy: 0.9996\n",
            "Epoch 31/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 8.5445e-04 - accuracy: 0.9998 - val_loss: 0.0019 - val_accuracy: 0.9995\n",
            "Epoch 32/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 9.3442e-04 - accuracy: 0.9997 - val_loss: 0.0016 - val_accuracy: 0.9997\n",
            "Epoch 33/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 7.1790e-04 - accuracy: 0.9998 - val_loss: 0.0016 - val_accuracy: 0.9996\n",
            "Epoch 34/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 7.2516e-04 - accuracy: 0.9998 - val_loss: 0.0016 - val_accuracy: 0.9996\n",
            "Epoch 35/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 9.1578e-04 - accuracy: 0.9997 - val_loss: 0.0020 - val_accuracy: 0.9994\n",
            "Epoch 36/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 6.7762e-04 - accuracy: 0.9998 - val_loss: 0.0015 - val_accuracy: 0.9997\n",
            "Epoch 37/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 6.5826e-04 - accuracy: 0.9998 - val_loss: 0.0015 - val_accuracy: 0.9996\n",
            "Epoch 38/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 7.5355e-04 - accuracy: 0.9998 - val_loss: 0.0016 - val_accuracy: 0.9996\n",
            "Epoch 39/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 7.1149e-04 - accuracy: 0.9998 - val_loss: 0.0015 - val_accuracy: 0.9996\n",
            "Epoch 40/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 5.8838e-04 - accuracy: 0.9999 - val_loss: 0.0024 - val_accuracy: 0.9994\n",
            "Epoch 41/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 5.2743e-04 - accuracy: 0.9999 - val_loss: 0.0015 - val_accuracy: 0.9997\n",
            "Epoch 42/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 7.9482e-04 - accuracy: 0.9998 - val_loss: 0.0019 - val_accuracy: 0.9995\n",
            "Epoch 43/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 6.0553e-04 - accuracy: 0.9998 - val_loss: 0.0013 - val_accuracy: 0.9997\n",
            "Epoch 44/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 6.0704e-04 - accuracy: 0.9998 - val_loss: 0.0020 - val_accuracy: 0.9995\n",
            "Epoch 45/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 6.3886e-04 - accuracy: 0.9998 - val_loss: 0.0015 - val_accuracy: 0.9996\n",
            "Epoch 46/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 5.7343e-04 - accuracy: 0.9998 - val_loss: 0.0013 - val_accuracy: 0.9997\n",
            "Epoch 47/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 4.2137e-04 - accuracy: 0.9999 - val_loss: 0.0018 - val_accuracy: 0.9996\n",
            "Epoch 48/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 6.0087e-04 - accuracy: 0.9998 - val_loss: 0.0016 - val_accuracy: 0.9997\n",
            "Epoch 49/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 5.2967e-04 - accuracy: 0.9999 - val_loss: 0.0012 - val_accuracy: 0.9998\n",
            "Epoch 50/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 5.2438e-04 - accuracy: 0.9998 - val_loss: 0.0012 - val_accuracy: 0.9997\n",
            "Epoch 51/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 5.0218e-04 - accuracy: 0.9998 - val_loss: 0.0018 - val_accuracy: 0.9996\n",
            "Epoch 52/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 6.3864e-04 - accuracy: 0.9998 - val_loss: 0.0020 - val_accuracy: 0.9995\n",
            "Epoch 53/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 6.7778e-04 - accuracy: 0.9998 - val_loss: 0.0016 - val_accuracy: 0.9996\n",
            "Epoch 54/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 4.3683e-04 - accuracy: 0.9999 - val_loss: 0.0017 - val_accuracy: 0.9996\n",
            "Epoch 55/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 7.2916e-04 - accuracy: 0.9998 - val_loss: 0.0018 - val_accuracy: 0.9996\n",
            "Epoch 56/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 4.6164e-04 - accuracy: 0.9999 - val_loss: 0.0017 - val_accuracy: 0.9996\n",
            "Epoch 57/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 4.3985e-04 - accuracy: 0.9999 - val_loss: 0.0022 - val_accuracy: 0.9995\n",
            "Epoch 58/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 4.4691e-04 - accuracy: 0.9999 - val_loss: 0.0015 - val_accuracy: 0.9996\n",
            "Epoch 59/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 6.0507e-04 - accuracy: 0.9998 - val_loss: 0.0010 - val_accuracy: 0.9998\n",
            "Epoch 60/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 4.2673e-04 - accuracy: 0.9999 - val_loss: 0.0027 - val_accuracy: 0.9995\n",
            "Epoch 61/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 3.7599e-04 - accuracy: 0.9999 - val_loss: 0.0019 - val_accuracy: 0.9996\n",
            "Epoch 62/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 6.4678e-04 - accuracy: 0.9998 - val_loss: 0.0024 - val_accuracy: 0.9995\n",
            "Epoch 63/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 3.3452e-04 - accuracy: 0.9999 - val_loss: 0.0013 - val_accuracy: 0.9997\n",
            "Epoch 64/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 4.5338e-04 - accuracy: 0.9999 - val_loss: 0.0022 - val_accuracy: 0.9995\n",
            "Epoch 65/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 5.2037e-04 - accuracy: 0.9998 - val_loss: 0.0013 - val_accuracy: 0.9997\n",
            "Epoch 66/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 3.9358e-04 - accuracy: 0.9999 - val_loss: 0.0014 - val_accuracy: 0.9997\n",
            "Epoch 67/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 4.2975e-04 - accuracy: 0.9999 - val_loss: 0.0018 - val_accuracy: 0.9997\n",
            "Epoch 68/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 3.7875e-04 - accuracy: 0.9999 - val_loss: 0.0012 - val_accuracy: 0.9997\n",
            "Epoch 69/100\n",
            "778/778 [==============================] - 3s 4ms/step - loss: 5.4449e-04 - accuracy: 0.9998 - val_loss: 0.0017 - val_accuracy: 0.9997\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "RSkN2iBlSDCM",
        "colab_type": "code",
        "outputId": "d1cdec13-b37f-419a-e2ac-48c98294db1a",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 72
        }
      },
      "source": [
        "results = model.evaluate(x_test, y_test, batch_size=5, verbose=1)\n",
        "print(\"Loss: %.2f\" % results[0])\n",
        "print(\"Acc: %.2f\" % results[1])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "34118/34118 [==============================] - 74s 2ms/step - loss: 0.0010 - accuracy: 0.9998\n",
            "Loss: 0.00\n",
            "Acc: 1.00\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LWqaG7eRSKiV",
        "colab_type": "code",
        "outputId": "1c1e4e81-515f-4dea-b9e2-ae451644b2fa",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 531
        }
      },
      "source": [
        "print(r.history.keys())\n",
        "plt.plot(r.history['loss'])\n",
        "plt.plot(r.history['val_loss'])\n",
        "plt.legend(['loss', 'val_loss'])\n",
        "plt.show()\n",
        "\n",
        "plt.plot(r.history['accuracy'])\n",
        "plt.plot(r.history['val_accuracy'])\n",
        "plt.legend(['accuracy', 'val_accuracy'])\n",
        "plt.show()"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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iSd2jScfNWoUAI8//jj33ntv13IjR47c67qXLFlCNBr0BDQ3N3PhhReyevVqzIxsNtu13r/9278lFovt8nyf/exn+e///m8uuuginnvuOe6+++5+2uL+E8qgT0QV9CJh8tRTT/H444/z3HPPUVFRweLFi5k7dy5vvvlmn9dROiyx+zj0ysrKrtvf+MY3OPnkk7n//vtZu3Ytixcv3uN6L7roIj7xiU+QTCZZsmRJ1xvBYBKqROw8jaDG/IqES3NzMyNHjqSiooI333yT559/nlQqxdNPP827774L0NV1c9ppp3Hbbbd1LdvZdVNXV8fKlSspFAp77ENvbm5mwoQJANx1111d00877TR++MMfdn1h2/l848ePZ/z48Xzzm9/koosu6r+N7kfhCvqszi4lEkann346uVyOadOmcc0113D88cczevRobr/9ds455xzmzJnDeeedB8DXv/51tm/fzsyZM5kzZw7Lli0D4Lvf/S5nnHEGH/zgBxk3blyvz3X11Vdz7bXXMm/evF1G4Vx88cVMnjyZ2bNnM2fOHO65556uxz7zmc8wadIkpk2bdpBegQNj7j7QNexi4cKFvnz58v1a9tqfv8bjK+v509c+0s9ViQxdK1euHLQBNlhcdtllzJs3jy984Qvvy/P19DcxsxfcfWFP8w++zqQD0Nl1IyLyflmwYAGVlZX88z//80CX0quQBb26bkTk/fXCCy8MdAl7FapUTGfz+rGUiEg34Qr6XEE/lhIR6SZUqahRNyIiuwtVKgZfxqrrRkSkVJ+C3sxON7NVZrbGzK7p4fHDzOwJM3vVzJ4ys4klj002s9+Y2UozW2FmU/qv/F3py1gRkd3tNRXNLArcBnwUmA5cYGbTu812M3C3u88GbgK+U/LY3cD33H0asAjY0h+F9yToo1eLXmSoq6qqGugSBpW+DK9cBKxx93cAzOxe4CxgRck804GrireXAb8ozjsdiLn7YwDu3tZPdfconc2TVIte5OD59TWw+bX+XefYWfDR7/bvOgeJXC43KI5905dUnACsL7m/oTit1CvAOcXbZwPDzKwGOBpoMrOfm9lLZva94ieEXZjZpWa23MyWNzQ07PtWFGnUjUg4XXPNNbscv+bGG2/km9/8Jqeeeirz589n1qxZ/PKXv+zTutra2npd7u677+46xMFnP/tZAOrr6zn77LOZM2cO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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_mQi5bR8SYMf",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "y_pred = model.predict(x_test)\n",
        "y_pred = np.round(y_pred, decimals=0).astype(int)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qeiyLpqHSas_",
        "colab_type": "code",
        "outputId": "dd3dcf3c-7a57-4d07-e3b8-2093b2632956",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 72
        }
      },
      "source": [
        "df_pred = pd.concat([pd.DataFrame(x_test), pd.DataFrame(y_test)], axis=1)\n",
        "df_pred.columns = df.drop('Time', axis=1).columns\n",
        "df_pred.rename(columns={\"Class\":\"Old_class\"}, inplace=True)\n",
        "df_pred['New_class'] = y_pred\n",
        "cm = pd.crosstab(df_pred[\"New_class\"], df_pred['Old_class'])\n",
        "true_pos = np.sum(np.diag(cm))\n",
        "false_pos = cm[0][1]\n",
        "false_neg = cm[1][0]\n",
        "precision = true_pos / (true_pos + false_pos) * 100\n",
        "recall = true_pos / (true_pos + false_neg) * 100\n",
        "f1 = 2 * (precision * recall) / (precision + recall)\n",
        "print(\"Precision: %.3f%%\" % (precision))\n",
        "print(\"Recall: %.3f%%\" % (recall))\n",
        "print(\"F1: %.3f%%\" % (f1))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Precision: 99.981%\n",
            "Recall: 100.000%\n",
            "F1: 99.991%\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "NjPd1aaeWlAE",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}