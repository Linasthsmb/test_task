import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")  # подавление предупреждений для чистоты вывода

# =========================================
# 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
# =========================================

def load_and_preprocess_data(path):
    # Загрузка CSV-файла с базой SKEMPI 2.0
    df = pd.read_csv(path, sep=';', comment='#', header=0, low_memory=False)

    # Явно задаём имена столбцов согласно документации SKEMPI 2.0
    df.columns = [
        'Pdb', 'Mutation(s)_PDB', 'Mutation(s)_cleaned', 'iMutation_Location(s)',
        'Hold_out_type', 'Hold_out_proteins', 'Affinity_mut (M)',
        'Affinity_mut_parsed', 'Affinity_wt (M)', 'Affinity_wt_parsed',
        'Reference', 'Protein 1', 'Protein 2', 'Temperature',
        'kon_mut (M^(-1)s^(-1))', 'kon_mut_parsed', 'kon_wt (M^(-1)s^(-1))',
        'kon_wt_parsed', 'koff_mut (s^(-1))', 'koff_mut_parsed',
        'koff_wt (s^(-1))', 'koff_wt_parsed', 'dH_mut (kcal mol^(-1))',
        'dH_wt (kcal mol^(-1))', 'dS_mut (cal mol^(-1) K^(-1))',
        'dS_wt (cal mol^(-1) K^(-1))', 'Notes', 'Method', 'SKEMPI version'
    ]

    # Обработка температуры — удаление текстовых примесей, приведение к float
    df['Temperature'] = df['Temperature'].astype(str).str.extract(r'(\d+\.?\d*)')
    df['Temperature'] = df['Temperature'].astype(float)

    # Целевая переменная — разность энтальпий (dH) между мутантом и диким типом
    df['ddG'] = df['dH_mut (kcal mol^(-1))'] - df['dH_wt (kcal mol^(-1))']

    # Удаление строк, где целевое значение ddG отсутствует
    df = df.dropna(subset=['ddG']).reset_index(drop=True)

    return df


# =========================================
# 2. ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛИ
# =========================================

def train_and_evaluate_model(df):
    # Отбор числовых признаков, исключая целевую переменную
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('ddG')
    X = df[numeric_cols]
    y = df['ddG']  # Задача — регрессия, предсказание ddG как непрерывной величины

    # Обработка пропущенных значений — медианное заполнение
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Масштабирование признаков — важно для большинства моделей
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

    # Разделение данных на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Используем XGBoost как мощный базовый алгоритм
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

    # Небольшой подбор гиперпараметров с помощью GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],          # количество деревьев
        'max_depth': [3, 5],                 # максимальная глубина дерева
        'learning_rate': [0.05, 0.1],        # скорость обучения
        'subsample': [0.8, 1.0]              # доля обучающей выборки для каждого дерева
    }

    grid_search = GridSearchCV(
        xgb, param_grid, cv=3, scoring='r2', verbose=1
    )
    grid_search.fit(X_train, y_train)

    # Извлечение модели с наилучшими параметрами
    best_model = grid_search.best_estimator_

    # Предсказания на тестовой выборке
    y_pred = best_model.predict(X_test)

    # Оценка модели на тесте
    print("\n=== МЕТРИКИ ===")
    print("MSE:", mean_squared_error(y_test, y_pred))     # среднеквадратичная ошибка
    print("R^2:", r2_score(y_test, y_pred))               # коэффициент детерминации

    # Дополнительно: оценка на всей выборке через кросс-валидацию
    scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='r2')
    print("\n=== Кросс-валидация R^2 ===")
    print("Среднее:", scores.mean())
    print("Std:", scores.std())

    # Визуализация: предсказания против истинных значений
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Настоящие ddG")
    plt.ylabel("Предсказанные ddG")
    plt.title("Предсказание vs Настоящие")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Визуализация важности признаков (feature importance)
    feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
    feature_importances.sort_values(ascending=True).plot(kind='barh', figsize=(10, 6))
    plt.title("Важность признаков (по XGBoost)")
    plt.tight_layout()
    plt.show()

    # ---- Матрица ошибок по знаку ddG ----
    # Преобразуем непрерывные значения в метки классов: 1 — положительный ddG, 0 — отрицательный или ноль
    y_test_bin = (y_test > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)

    cm = confusion_matrix(y_test_bin, y_pred_bin)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["≤ 0", "> 0"], yticklabels=["≤ 0", "> 0"])
    plt.xlabel("Предсказанный знак ddG")
    plt.ylabel("Истинный знак ddG")
    plt.title("Матрица ошибок по знаку ddG")
    plt.tight_layout()
    plt.show()



# =========================================
# 3. ЗАПУСК СКРИПТА
# =========================================

if __name__ == "__main__":
    path = "skempi_v2.csv"  # путь к CSV-файлу базы данных SKEMPI 2.0
    df = load_and_preprocess_data(path)  # загрузка и очистка
    train_and_evaluate_model(df)         # обучение и визуализация
