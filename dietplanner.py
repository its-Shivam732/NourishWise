import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import random

class DietTypeRecommender:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        self.mlp_model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def balance_diet_types(self, data):
        """
        Balance the dataset by ensuring equal number of samples for each diet type
        by randomly removing excess samples.
        """
        diet_counts = data['Diet_type'].value_counts()
        min_count = diet_counts.min()

        balanced_data = pd.DataFrame()
        for diet_type in diet_counts.index:
            diet_samples = data[data['Diet_type'] == diet_type]
            balanced_samples = diet_samples.sample(n=min_count, random_state=42)
            balanced_data = pd.concat([balanced_data, balanced_samples])

        balanced_data = balanced_data.reset_index(drop=True)
        return balanced_data

    def generate_objective_column(self, data):
        """
        Generate the 'objective' column based on the 'Diet_type' with outliers (10% of values)
        """
        objective_mapping = {
            'mediterranean': 7,  # Increase Weight
            'keto': 8,           # Increase Muscle
            'paleo': 9,          # Maintain Weight
            'dash': 10           # Reduce Weight
        }

        data['objective'] = data['Diet_type'].map(objective_mapping)

        # Add 10% outliers randomly
        outlier_indices = random.sample(range(len(data)), k=int(0.1 * len(data)))
        for idx in outlier_indices:
            data.at[idx, 'objective'] = random.choice([7, 8, 9, 10])

        return data

    def calculate_macros(self, height, weight, objective):
        """
        Calculate optimal macronutrients based on height, weight, and objective.
        """
        height_m = height / 100  # convert to meters
        bmi = weight / (height_m ** 2)

        # Calculate Base Metabolic Rate (BMR) using Harris-Benedict equation
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * 30)

        daily_calories = bmr * 1.55  # Assuming moderate activity

        # Adjust based on the objective
        if objective == 7:  # Increase Weight
            daily_protein = weight * 2.0
            daily_fat = (daily_calories * 0.25) / 9
            daily_carbs = (daily_calories - (daily_protein * 4) - (daily_fat * 9)) / 4
        elif objective == 8:  # Increase Muscle
            daily_protein = weight * 2.2
            daily_fat = (daily_calories * 0.30) / 9
            daily_carbs = (daily_calories - (daily_protein * 4) - (daily_fat * 9)) / 4
        elif objective == 9:  # Maintain Weight
            daily_protein = weight * 1.8
            daily_fat = (daily_calories * 0.30) / 9
            daily_carbs = (daily_calories - (daily_protein * 4) - (daily_fat * 9)) / 4
        elif objective == 10:  # Reduce Weight
            daily_protein = weight * 1.5
            daily_fat = (daily_calories * 0.20) / 9
            daily_carbs = (daily_calories - (daily_protein * 4) - (daily_fat * 9)) / 4

        return {
            'protein': round(daily_protein / 3, 1),
            'carbs': round(daily_carbs / 3, 1),
            'fat': round(daily_fat / 3, 1)
        }

    def prepare_data(self, dataset):
        # Encode diet types using label_encoder
        self.label_encoder.fit(dataset['Diet_type'])

        # Encode objectives using objective_encoder
        self.objective_encoder = LabelEncoder()
        self.objective_encoder.fit(dataset['objective'])

        # Prepare features and target
        X = dataset[['Protein(g)', 'Carbs(g)', 'Fat(g)', 'objective']].values
        y = self.label_encoder.transform(dataset['Diet_type'])  # Corrected to use label_encoder

        # Scale features
        X = self.scaler.fit_transform(X)

        return X, y

    def train_models(self, X, y):
        """
        Train both Random Forest and MLP models
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest
        self.rf_model.fit(X_train, y_train)

        # Train MLP
        self.mlp_model.fit(X_train, y_train)

        return X_test, y_test

    def validate_models(self, X_test, y_test):
        """
        Validate models using k-fold cross validation
        """
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        metrics = {
            'Random Forest': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
            'MLP': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        }

        for train_idx, val_idx in kf.split(X_test):
            X_train_fold, X_val_fold = X_test[train_idx], X_test[val_idx]
            y_train_fold, y_val_fold = y_test[train_idx], y_test[val_idx]

            rf_pred = self.rf_model.predict(X_val_fold)
            metrics['Random Forest']['accuracy'].append(accuracy_score(y_val_fold, rf_pred))
            metrics['Random Forest']['precision'].append(precision_score(y_val_fold, rf_pred, average='weighted'))
            metrics['Random Forest']['recall'].append(recall_score(y_val_fold, rf_pred, average='weighted'))
            metrics['Random Forest']['f1'].append(f1_score(y_val_fold, rf_pred, average='weighted'))

            mlp_pred = self.mlp_model.predict(X_val_fold)
            metrics['MLP']['accuracy'].append(accuracy_score(y_val_fold, mlp_pred))
            metrics['MLP']['precision'].append(precision_score(y_val_fold, mlp_pred, average='weighted'))
            metrics['MLP']['recall'].append(recall_score(y_val_fold, mlp_pred, average='weighted'))
            metrics['MLP']['f1'].append(f1_score(y_val_fold, mlp_pred, average='weighted'))

        return metrics

    def get_diet_recommendation(self, height, weight, objective):
        """
        Get diet type recommendation based on height, weight, and objective
        """
        macros = self.calculate_macros(height, weight, objective)

        X_pred = np.array([[macros['protein'], macros['carbs'], macros['fat'], objective]])
        X_pred_scaled = self.scaler.transform(X_pred)

        rf_pred_proba = self.rf_model.predict_proba(X_pred_scaled)[0]
        mlp_pred_proba = self.mlp_model.predict_proba(X_pred_scaled)[0]

        combined_proba = (rf_pred_proba + mlp_pred_proba) / 2

        top_indices = np.argsort(combined_proba)[-3:][::-1]
        recommendations = []

        for idx in top_indices:
            objective_value = self.objective_encoder.inverse_transform([idx])[0]
            confidence = combined_proba[idx]
            recommendations.append({
                'diet_type': self.label_encoder.inverse_transform([idx])[0],
                'objective': objective_value,
                'confidence': round(confidence * 100, 2),
                'macros_per_meal': macros,
                'macros_per_day': {
                    'protein': macros['protein'] * 3,
                    'carbs': macros['carbs'] * 3,
                    'fat': macros['fat'] * 3
                }
            })

        return recommendations

    def filter_recipes_by_top_diet_type(self, dataset, top_diet_type):
        """
        Filter recipes based on the top diet type with the highest confidence.
        """
        filtered_data = dataset[dataset['Diet_type'] == top_diet_type]
        return filtered_data

    def prepare_knn_data(self, dataset):
        """
        Prepare data for k-NN.
        """
        X = dataset[['Protein(g)', 'Carbs(g)', 'Fat(g)']].values
        y = dataset['Recipe_name'].values
        return X, y

    def predict_recipes(self, knn_model, X):
        """
        Predict recipe names based on the features.
        """
        predicted_recipes = knn_model.predict(X)
        return predicted_recipes

    def get_recipe_cluster(self, data, recipe_name):
        """
        Given a recipe name, retrieve its associated cluster.
        """
        recipe_data = data[data['Recipe_name'] == recipe_name]
        if not recipe_data.empty:
            return recipe_data['Cluster'].iloc[0]
        else:
            return None

    def get_recipes_in_same_cluster(self, data, cluster_number):
        """
        Given a cluster number, return all recipes in that cluster.
        """
        recipes_in_cluster = data[data['Cluster'] == cluster_number]
        return recipes_in_cluster

    def apply_kmeans_clustering(self, filtered_data, n_clusters=14):
        """
        Apply K-Means clustering based on Protein(g), Carbs(g), and Fat(g).
        """
        X = filtered_data[['Protein(g)', 'Carbs(g)', 'Fat(g)']].values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        filtered_data['Cluster'] = kmeans.fit_predict(X)
        return filtered_data

    def plot_validation_metrics(self,metrics):
        """
        Plot cross-validation metrics for each model.
        """
        models = metrics.keys()
        metrics_list = ['accuracy', 'precision', 'recall', 'f1']

        # Create subplots for metrics
        fig, axes = plt.subplots(1, len(metrics_list), figsize=(20, 5))
        fig.suptitle('Cross-Validation Metrics for Models', fontsize=16)

        for i, metric_name in enumerate(metrics_list):
            for model in models:
                axes[i].plot(metrics[model][metric_name], label=model, marker='o')

            axes[i].set_title(metric_name.capitalize())
            axes[i].set_xlabel('Fold')
            axes[i].set_ylabel(metric_name.capitalize())
            axes[i].legend()
            axes[i].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # Example usage
    def getMeal(self, height, weight, preference, goal):
        # Sample dataset (with per-meal values)
        data = pd.read_csv('All_Diets.csv')
        data = data[data['Diet_type'] != 'vegan']

        # Initialize recommender
        balanced_data = self.balance_diet_types(data)
        balanced_data = self.generate_objective_column(balanced_data)

        # Prepare and train models
        X, y = self.prepare_data(balanced_data)
        X_test, y_test = self.train_models(X, y)

        # Validate models
        metrics = self.validate_models(X_test, y_test)
        # self.plot_validation_metrics(metrics)
        print("\nModel Validation Metrics:")
        for model, model_metrics in metrics.items():
            print(f"\n{model}:")
            for metric, values in model_metrics.items():
                # Calculate the mean of the list of metric values
                mean_value = np.mean(values)
                print(f"{metric}: {mean_value:.4f}")

        # Example recommendation
        objective_mapping = {
            'Increase Weight': 7,
            'Increase Muscle': 8,
            'Maintain Weight': 9,
            'Reduce Weight': 10
        }

        # Get objective based on goal
        if goal not in objective_mapping:
            return {"error": "Invalid goal"}

        objective = objective_mapping[goal]
        recommendations = self.get_diet_recommendation(height, weight, objective)

        print("\nDiet Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['diet_type'].upper()} (Objective: {rec['objective']} | Confidence: {rec['confidence']}%)")
            print("Recommended macros per meal:")
            print(f"Protein: {rec['macros_per_meal']['protein']}g")
            print(f"Carbs: {rec['macros_per_meal']['carbs']}g")
            print(f"Fat: {rec['macros_per_meal']['fat']}g")
            print("\nDaily totals:")
            print(f"Protein: {rec['macros_per_day']['protein']}g")
            print(f"Carbs: {rec['macros_per_day']['carbs']}g")
            print(f"Fat: {rec['macros_per_day']['fat']}g")
        top_diet_type = recommendations[0]['diet_type']

        macros_per_meal = recommendations[0]['macros_per_meal']
        X_knnvalue = np.array([[macros_per_meal['protein'], macros_per_meal['carbs'], macros_per_meal['fat']]])

        # Filter recipes based on the top diet type
        filtered_data = self.filter_recipes_by_top_diet_type(pd.read_csv('All_Diets.csv'), top_diet_type)
        # Prepare data for k-NN prediction
        X_knn, y_knn = self.prepare_knn_data(filtered_data)
        # Train k-NN model
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X_knn, y_knn)

        # Predict the recipe names
        predicted_recipe = self.predict_recipes(knn_model, X_knnvalue)[0]
        predicted_recipe_row = filtered_data[filtered_data['Recipe_name'] == predicted_recipe]

        if predicted_recipe_row.empty:
            return {"error": "Recipe not found."}

        # Get the required columns
        recipe_info = predicted_recipe_row[['Diet_type', 'Recipe_name', 'Protein(g)', 'Carbs(g)', 'Fat(g)']].to_dict(
            orient='records')

        # Return the recipe information in JSON format
        return {"predicted_recipe": recipe_info[0]}

        # # Display predictions
        # print(predicted_recipe)
        #
        # predicted_cluster = self.get_recipe_cluster(clustered_data, predicted_recipe)
        # print(f"Predicted Recipe Cluster: {predicted_cluster}")
        #
        # # Fetch all recipes in the same cluster
        # if predicted_cluster is not None:
        #     recipes_in_cluster = self.get_recipes_in_same_cluster(clustered_data, predicted_cluster)
        #     print(f"\nRecipes in the same cluster as {predicted_recipe}:")
        #     print(recipes_in_cluster[['Recipe_name', 'Protein(g)', 'Carbs(g)', 'Fat(g)', 'Cluster']])
        # else:
        #     print("Cluster not found for the predicted recipe.")

    def getMoreMoreMealRecommendation(self, Diet_type,	Recipe_name):
        filtered_data = self.filter_recipes_by_top_diet_type(pd.read_csv('All_Diets.csv'), Diet_type)
        clustered_data = self.apply_kmeans_clustering(filtered_data)
        predicted_cluster = self.get_recipe_cluster(clustered_data, Recipe_name)
        print(f"Predicted Recipe Cluster: {predicted_cluster}")

        # Fetch all recipes in the same cluster
        if predicted_cluster is not None:
            recipes_in_cluster = self.get_recipes_in_same_cluster(clustered_data, predicted_cluster)
            print(f"\nRecipes in the same cluster as {Recipe_name}:")
            print(recipes_in_cluster[['Recipe_name', 'Protein(g)', 'Carbs(g)', 'Fat(g)', 'Cluster']])
            recipes_in_cluster = recipes_in_cluster[['Recipe_name', 'Protein(g)', 'Carbs(g)', 'Fat(g)']]

            # Convert to a list of dictionaries to return as JSON
            recommendations_list = recipes_in_cluster.to_dict(orient='records')
            return recommendations_list
        else:
            print("Cluster not found for the predicted recipe.")

