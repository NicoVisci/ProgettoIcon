from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
import pandas as pds


class TrainingManager:
    # Class constructor
    def __init__(self, input_file_path):
        self.file_path = input_file_path
        self.data = None

        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

    # Data loaded directly from class specified file, assuming the file format is CSV
    # Can catch specific FileNotFound exceptions or generic ones
    def load_data(self):
        # Trying to load the specified CSV file in a pandas DataFrame
        try:
            self.data = pds.read_csv(self.file_path, low_memory=False)
            print("Caricamento dati completato")
        except FileNotFoundError:
            print(f"Errore: File non trovato - {self.file_path}, probabile file mancante oppure il percorso indicato è errato")
        except Exception as e:
            print(f"Errore generico durante il caricamento dei dati: {str(e)}")

    def learning_and_test(self):
        Y = pds.DataFrame(self.data['price'])

        columns_to_drop = ['price', 'host_identity_verified', 'neighbourhood group', 'lat', 'long', 'last review', 'cancellation_policy', 'availability 365', 'instant_bookable']
        X = self.data.drop(columns=columns_to_drop, axis=1)

        le = LabelEncoder()
        X['neighbourhood'] = le.fit_transform(X['neighbourhood'])
        X['room type'] = le.fit_transform(X['room type'])

        # Initialize MinMaxScaler
        scaler = MinMaxScaler()
        # Fit and transform the data
        # X.scaled = scaler.fit_transform(X)
        # Y.scaled = scaler.fit_transform(Y)

        # Convert the scaled data back to a DataFrame
        X_scaled = X #pds.DataFrame(scaler.fit_transform(X), columns=X.columns)
        Y_scaled = Y #pds.DataFrame(scaler.fit_transform(Y), columns=Y.columns)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split (X_scaled, Y_scaled['price'], test_size = 0.2, random_state = 42)

        # Initialize the Random Forest Regressor with the best hyperparameters
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

        # Set up k-fold cross-validation
        k_folds = 10
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        print("K-fold set up, starting evaluating the MSE scores")

        # Perform k-fold cross-validation
        mse_scores = cross_val_score(rf_regressor, self.X_train, self.Y_train, cv=kf, scoring='neg_mean_squared_error')

        print("Starting evaluating the R2 scores")
        r2_scores = cross_val_score(rf_regressor, self.X_train, self.Y_train, cv=kf, scoring='r2')
        # Convert MSE scores to positive values
        mse_scores = -mse_scores

        # Print results
        print(f"K-Fold Cross-Validation Results (k={k_folds}):")
        print(f"Mean Squared Error: {mse_scores.mean():.4f} (+/- {mse_scores.std() * 2:.4f})")
        print(f"R-squared Score: {r2_scores.mean():.4f} (+/- {r2_scores.std() * 2:.4f})")

        '''
        # If you want to see individual fold scores
        print("\nIndividual Fold Scores:")
        for fold, (mse, r2) in enumerate(zip(mse_scores, r2_scores), 1):
            print(f"Fold {fold}: MSE = {mse:.4f}, R2 = {r2:.4f}")
        '''
        print("Starting the training process")
        # Train final model on entire dataset
        rf_regressor.fit(self.X_train, self.Y_train)

        print("Starting the testing process")
        # Make predictions
        Y_pred = rf_regressor.predict(self.X_test)

        # Calculate final metrics
        final_mse = mean_squared_error(self.Y_test, Y_pred)
        final_r2 = r2_score(self.Y_test, Y_pred)

        print("\nFinal Model Performance (trained on entire dataset):")
        print(f"Mean Squared Error: {final_mse:.4f}")
        print(f"R-squared Score: {final_r2:.4f}")


        # Inverse transform
        # self.Y_test = pds.DataFrame(scaler.inverse_transform(Y), columns=Y.columns)
        # Y_pred = pds.DataFrame(scaler.inverse_transform(Y), columns=Y.columns)

        plt.figure(figsize=(10, 6))

        plt.scatter(Y_pred, self.Y_test, alpha=0.6)

        # Customize the plot
        plt.title('Actual Price vs Predicted Price', fontsize=14)
        plt.xlabel('Actual Price', fontsize=12)
        plt.ylabel('Predicted Price', fontsize=12)

        # Set the axis limits
        plt.xlim(0, 1300)
        plt.ylim(0, 1300)

        # Add gridlines
        plt.grid(True, linestyle='--', alpha=0.7)

        # Show the plot
        plt.tight_layout()
        plt.show()
        '''

def call():
    original_file_path = '../data/Post_PreProcessing/Airbnb_Processed_Data.csv'
    analyzer = TrainingManager(original_file_path)
    analyzer.load_data()
    analyzer.learning_and_test()