from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

class TrafficPredictor:
    def __init__(self, model_path="data/traffic_model.pkl"):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_path = model_path

    def train(self, data_file="data/traffic_data.csv"):
        df = pd.read_csv(data_file)
        X = df[["node_from", "node_to", "hour", "day_of_week"]]
        y = df["congestion_factor"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        score = self.model.score(X_test, y_test)
        print(f"Model R² Score: {score}")
        import pickle
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

    def predict(self, node_from, node_to, hour, day_of_week):
        X = [[node_from, node_to, hour, day_of_week]]
        return self.model.predict(X)[0]

def main():
    predictor = TrafficPredictor()
    predictor.train()

if __name__ == "__main__":
    main()