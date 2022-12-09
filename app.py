import streamlit as st
import pandas as pd

view = [100,150,30]
st.write('# YOUTUBE viwew')
st.write('# YOUTUBE viwew')

st.write('# YOUTUBE viwew')

st.write('## raw')

view 
st.write('## raw')
st.write('## bar chart')

st.bar_chart(view)

sview = pd.Series(view)
sview
sview
sview
sview

# dd


#model part
    import pandas as pd
                import matplotlib.pyplot as plt
                import numpy as np
                from sklearn.linear_model import LinearRegression
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import PolynomialFeatures

                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import Pipeline
                from sklearn.model_selection import train_test_split

                from sklearn.preprocessing import StandardScaler

                from matplotlib import rc
                rc('font', family='AppleGothic')

                plt.rcParams['axes.unicode_minus'] = False

                from sklearn.preprocessing import MinMaxScaler


                from scipy import stats





                import pandas as pd
                df = pd.read_excel("5. 북한이탈주민.xlsx")
                df.head()
                # df.info()




                df_X = df.drop(columns = ['Q19A1','Q19A2','Q19A3','Q19A4','Q19A5','Q19A6','Q19A7','Q19A8','Q20A1','Q20A2','Q20A3','Q20A4','Q20A5'])  #여러개 열을 삭제할 때.


                df_X = df_X.drop('ID',axis=1)
                df_X.columns.values




                df_X= df_X.dropna(axis=1).values
                X = df_X




                y = df['Q20A1'].values
                y




                X_train,X_test,y_train,y_test =  train_test_split(X,y,test_size = 0.2)

                model = LinearRegression()
                model.fit(X_train,y_train)

                y_predict = model.predict(X_test)

                # scaler
                scaler = MinMaxScaler(feature_range=(1,4)) ## 각 칼럼 데이터 값을1~ 4범위로 변환
                y_predict = scaler.fit_transform(y_predict.reshape(-1, 1)) ##각 feature의 최솟값과 최댓값을 기준으로 1~4 구간 내에 균등하게 값을 배정
                print(y_predict)
                # np.round
                y_predict = np.round(y_predict)

                r, p = stats.pearsonr(y_test.flatten(), y_predict.flatten())

                print(f'Pearson correlation coefficient is {r}')
                print(f'r^2 is {r**2}')

                plt.scatter(y_test, y_predict, alpha=0.4)
                plt.xlabel("Actual Happiness")
                plt.ylabel("predicted Happiness")
                plt.title("행복도 예측")
                plt.show()

                plt.plot(y_test, 'o',label = 'y_test' )
                plt.plot(y_predict,'o', label = 'y_predict')

                plt.legend(loc = 'upper center')

                plt.show()
#modeln Fin

view = [100,150,30]

st.write(r)
st.bar_chart(view)

sview
sview
sview
