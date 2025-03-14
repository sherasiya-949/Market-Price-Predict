# from flask import Flask, request, jsonify, render_template
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression

# app = Flask(__name__)


# # Load the dataset
# df = pd.read_csv("Price_Agriculture_commodities_Week.csv")
# df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], errors='coerce')
# df.dropna(subset=['Arrival_Date'], inplace=True)

# def get_unique(column):
#     return df[column].unique().tolist()

# # Home Route
# @app.route('/')
# def index():
#     return render_template('index.html')  # Ensure 'index.html' is in the 'templates' folder

# @app.route('/get_states', methods=['GET'])
# def get_states():
#     states = get_unique('State')
#     return jsonify(states)

# @app.route('/get_districts', methods=['GET'])
# def get_districts():
#     state = request.args.get('state')
#     districts = df[df['State'] == state]['District'].unique().tolist()
#     return jsonify(districts)

# @app.route('/get_markets', methods=['GET'])
# def get_markets():
#     district = request.args.get('district')
#     markets = df[df['District'] == district]['Market'].unique().tolist()
#     return jsonify(markets)

# @app.route('/get_data', methods=['GET'])
# def get_data():
#     state = request.args.get('state')
#     district = request.args.get('district')
#     market = request.args.get('market')

#     data = df[(df['State'] == state) & (df['District'] == district) & (df['Market'] == market)]
#     result = data[['Commodity', 'Variety', 'Grade', 'Min Price', 'Max Price', 'Modal Price', 'Arrival_Date']].to_dict(orient='records')
#     return jsonify(result)

# @app.route('/get_commodities', methods=['GET'])
# def get_commodities():
#     market = request.args.get('market')
#     commodities = df[df['Market'] == market]['Commodity'].unique().tolist()
#     return jsonify(commodities)

# # ✅ Prediction Route
# @app.route('/predict_price', methods=['POST'])
# def predict_price():
#     data = request.get_json()
#     min_price = float(data['min_price'])
#     max_price = float(data['max_price'])
#     commodity = data.get('commodity')

#     if commodity:
#         filtered_df = df[df['Commodity'] == commodity]
#         if not filtered_df.empty:
#             # Train model on commodity-specific data
#             X = filtered_df[['Min Price', 'Max Price']].values
#             y = filtered_df['Modal Price'].values
            
#             model = LinearRegression()
#             model.fit(X, y)

#             # Predict modal price
#             X_input = np.array([[min_price, max_price]])
#             predicted_price = model.predict(X_input)[0]

#             return jsonify({'predicted_modal_price': predicted_price})

#     return jsonify({'error': 'No matching data for prediction'})

# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/data-view')
# def data_view():
#     data = df[['Commodity', 'Variety', 'Grade', 'Min Price', 'Max Price', 'Modal Price', 'Arrival_Date']].to_dict(orient='records')
#     return render_template('data-view.html', data=data)

# @app.route('/contact', methods=['GET', 'POST'])
# def contact():
#     if request.method == 'POST':
#         name = request.form.get('name')
#         email = request.form.get('email')
#         message = request.form.get('message')
#         print(f"Message from {name} ({email}): {message}")
#         return jsonify({'message': 'Thank you for your message!'})

#     return render_template('contact.html')

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("Price_Agriculture_commodities_Week.csv")
df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], errors='coerce')
df.dropna(subset=['Arrival_Date'], inplace=True)

def get_unique(column):
    return df[column].unique().tolist()

# Home Route
@app.route('/')
def index():
    return render_template('index.html')  # Ensure 'index.html' is in the 'templates' folder

@app.route('/get_states', methods=['GET'])
def get_states():
    states = get_unique('State')
    return jsonify(states)

@app.route('/get_districts', methods=['GET'])
def get_districts():
    state = request.args.get('state')
    districts = df[df['State'] == state]['District'].unique().tolist()
    return jsonify(districts)

@app.route('/get_markets', methods=['GET'])
def get_markets():
    district = request.args.get('district')
    markets = df[df['District'] == district]['Market'].unique().tolist()
    return jsonify(markets)

@app.route('/get_data', methods=['GET'])
def get_data():
    state = request.args.get('state')
    district = request.args.get('district')
    market = request.args.get('market')

    data = df[(df['State'] == state) & (df['District'] == district) & (df['Market'] == market)]
    result = data[['Commodity', 'Variety', 'Grade', 'Min Price', 'Max Price', 'Modal Price', 'Arrival_Date']].to_dict(orient='records')
    return jsonify(result)

@app.route('/get_commodities', methods=['GET'])
def get_commodities():
    market = request.args.get('market')
    commodities = df[df['Market'] == market]['Commodity'].unique().tolist()
    return jsonify(commodities)

# ✅ Custom Linear Regression Function
def train_linear_regression(X, y):
    X = np.c_[np.ones(X.shape[0]), X]  # Add bias term (intercept)
    
    # theta = (X'X)^-1 * X'Y
    theta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    return theta

def predict_linear_regression(X, theta):
    X = np.c_[np.ones(X.shape[0]), X]  # Add bias term (intercept)
    return X @ theta

# ✅ Prediction Route
@app.route('/predict_price', methods=['POST'])
def predict_price():
    data = request.get_json()
    min_price = float(data['min_price'])
    max_price = float(data['max_price'])
    commodity = data.get('commodity')

    if commodity:
        filtered_df = df[df['Commodity'] == commodity]
        if not filtered_df.empty:
            # Train model on commodity-specific data
            X = filtered_df[['Min Price', 'Max Price']].values
            y = filtered_df['Modal Price'].values

            # Train custom linear regression
            theta = train_linear_regression(X, y)

            # Predict modal price
            X_input = np.array([[min_price, max_price]])
            predicted_price = predict_linear_regression(X_input, theta)[0]

            return jsonify({'predicted_modal_price': predicted_price})

    return jsonify({'error': 'No matching data for prediction'})

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/data-view')
def data_view():
    data = df[['Commodity', 'Variety', 'Grade', 'Min Price', 'Max Price', 'Modal Price', 'Arrival_Date']].to_dict(orient='records')
    return render_template('data-view.html', data=data)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        print(f"Message from {name} ({email}): {message}")
        return jsonify({'message': 'Thank you for your message!'})

    return render_template('contact.html')

# if __name__ == '__main__':
#     app.run(debug=True, use_reloader=False)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use port from environment if available
    app.run(host="0.0.0.0", port=port)
