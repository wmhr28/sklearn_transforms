from sklearn.base import BaseEstimator, TransformerMixin


# All sklearn Transforms must have the `transform` and `fit` methods
class CustomRandomUnderSampling(BaseEstimator, TransformerMixin): 
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        count_class_0, count_class_1 = data['OBJETIVO'].value_counts()

        # Divide by class
        df_class_0 = data[data['OBJETIVO'] == 'Aceptado']
        df_class_1 = data[data['OBJETIVO'] == 'Sospechoso']
        
        #Creamos en dataframe con data ejemplo
        df_class_0_under = df_class_0.sample(count_class_1)
        df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)
        
        
        # Devolvemos un nuevo dataframe de datos
        return df_test_under

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
