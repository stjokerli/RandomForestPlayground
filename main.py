import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import sys
import os
import io
import time
import pickle
import base64



# df.info()表示の関数
def Df_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    for line in buffer.getvalue().split('\n'):
        st.text(line)

        
        
# CSV読み込みの関数
def Load_data(file):
    # CSVをdfに取り込み
    df = pd.read_csv(file)

    # dfを表示
    st.write('読み込んだファイル：')
    st.write(df)
    st.write('{}行、{}列のデータを読み込みました。'.format(df.shape[0], df.shape[1]))

    # df.info()を表示
    st.write('読み込んだファイルの情報：')
    Df_info(df)

    return df



def download_link(dl_type, dl_object, dl_filename, dl_link_text):

    if dl_type == 'pd':
        dl_object = dl_object.to_csv()
        b64 = base64.b64encode(dl_object.encode('cp932')).decode('cp932')
        return f'<a href="data:file/csv;base64,{b64}" download="{dl_filename}">{dl_link_text}</a>'
    elif dl_type == 'fig':
        fig_BytesIO = io.BytesIO()
        dl_object.savefig(fig_BytesIO, bbox_inches='tight', format='pdf')
        fig_BytesIO.seek(0)
        b64 = base64.b64encode(fig_BytesIO.read())
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{dl_filename}">{dl_link_text}</a>'
    elif dl_type == 'pkl':
        pkl_BytesIO = io.BytesIO()
        pickle.dump(dl_object, pkl_BytesIO)
        pkl_BytesIO.seek(0)
        b64 = base64.b64encode(pkl_BytesIO.read())
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{dl_filename}">{dl_link_text}</a>'

    

flg_impute = False
flg_prep = False
flg_fit = False
result_df = pd.DataFrame(columns=['Train_Score', 'Test_Score', 'Memo'], index=[])

st.title('Interactive Random Forest Classifier & Regressor')
st.write('汎用的なデータに対してRandom Forestを用いた回帰・分類の学習および予測をブラウザー上で行うプログラムです。')

st.header('学習データの読み込み')
st.write('左のメニューから、学習させるデータを選んでください。')


# ファイルアップロード
st.sidebar.subheader('学習データの読み込み')
train_file = st.sidebar.file_uploader('学習ファイルのアップロード', type='csv')

if train_file is not None:
    
    # アップロードファイルをメイン画面にデータ表示
    train_df = Load_data(train_file)

# Correlation Mapを表示

    if st.sidebar.button('Correlation Mapを表示する'):
        fig_corr = plt.figure()
        sns.heatmap(train_df.corr(), annot=True, fmt='1.1f', linewidths=0.5, center=0)
        st.header('Correlation Mapを表示')
        st.pyplot(fig_corr)

        tmp_download_link = download_link('fig', fig_corr, 'corr_heatmap.pdf', 'PDFファイルをダウンロード')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

# Pairplotを表示

    if st.sidebar.button('Pair Plotを表示する'):
        fig_pair = sns.pairplot(train_df)
        st.header('Pair Plotを表示')
        st.pyplot(fig_pair)
        
        tmp_download_link = download_link('fig', fig_pair, 'pairplot.pdf', 'PDFファイルをダウンロード')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

# 学習データの準備

    st.header('学習データの準備')
    st.write('左のメニューから、予測から除く列と予測対象の列を選び、次の処理に進んでください。')

    st.sidebar.subheader('学習データの準備')
    drop_columns = st.sidebar.multiselect('不要な列を選択', train_df.columns)
    y_column = st.sidebar.selectbox('予測対象の列を選択', train_df.columns, index=len(train_df.columns)-1)

    prep = []
    
    
# 欠損値の処理
    
    st.sidebar.subheader('欠損値の処理')
    if st.sidebar.checkbox('欠損値の処理をする'):
        
        flg_impute = True
        
        st.header('欠損値の処理')
        st.write('左のメニューから、処理を選択してください。')

        X = train_df.drop(drop_columns, axis=1)
        y = train_df[y_column].astype(str)

        st.write('処理前のXの情報：')
        Df_info(X)
        st.write('欠損値のある列：')
        st.write(X.dtypes[X.isnull().any()])
        
        impute_list = ['Drop', 'Impute by Mean', 'Impute by Median', 'Impute by Most Frequent']

        impute_sel = st.sidebar.multiselect('欠損値の処理をすべて選択', impute_list)
        
        impute_drop = []
        impute_med = []
        impute_mean = []
        impute_freq = []
        
        if 'Drop' in impute_sel:
            impute_drop = st.sidebar.multiselect('欠損行をDropする列を選択', X.columns)
        
        if 'Impute by Median' in impute_sel:
            impute_med = st.sidebar.multiselect('欠損値を中央値で補完する列を選択', X.columns)
        
        if 'Impute by Mean' in impute_sel:
            impute_mean = st.sidebar.multiselect('欠損値を平均値で補完する列を選択', X.columns)
        
        if 'Impute by Most Frequent' in impute_sel:
            impute_freq = st.sidebar.multiselect('欠損値を最頻値で補完する列を選択', X.columns)
        
        if st.sidebar.checkbox('欠損値の処理を実行'):
            if impute_drop:
                train_df.dropna(axis=0, subset=impute_drop, inplace=True)
                X = train_df.drop(drop_columns, axis=1)
                y = train_df[y_column].astype(str)
            if impute_mean:
                pipe_mean = make_pipeline(SimpleImputer(strategy='mean'))
                X[impute_mean] = pipe_mean.fit_transform(X[impute_mean])
            if impute_med:
                pipe_med = make_pipeline(SimpleImputer(strategy='median'))
                X[impute_med] = pipe_med.fit_transform(X[impute_med])
            if impute_freq:
                pipe_freq = make_pipeline(SimpleImputer(strategy='most_frequent'))
                X[impute_freq] = pipe_freq.fit_transform(X[impute_freq])
                
            st.write('処理後のXの情報：')
            Df_info(X)
            
            if X.isnull().values.sum() > 0:
                st.write('まだ欠損値があります。戻ってやり直してください。')

            # Xの保存
            tmp_download_link = download_link('pd', X, 'X.csv', '学習データXのCSVファイルをダウンロード')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

            st.write('次の処理に進んでください。')
            
# 前処理
    st.sidebar.subheader('前処理')
    if st.sidebar.checkbox('前処理を実行する'):
        
        flg_prep = True

        st.header('前処理')
        st.write('左のメニューから、処理を選択してください。')

        if not flg_impute:
            X = train_df.drop(drop_columns, axis=1)
            y = train_df[y_column].astype(str)

        # ダミー化
        if len(X.select_dtypes(include=object)) > 0:
            X = pd.get_dummies(X)

        st.write('処理前のXの情報：')
        Df_info(X)
        
        X_len = X.shape[1]
        PCA__n_components = X_len
        
        # 前処理の辞書
        prep_dict = {'PCA': PCA(n_components = PCA__n_components),
                     'Standard Scaler': StandardScaler(),
                     'Min Max Scaler': MinMaxScaler(),
                     'Robust Scaler': RobustScaler()}
        
        # 前処理を選択
        prep_sel = st.sidebar.multiselect('前処理をすべて選択', list(prep_dict.keys()))

        if 'PCA' in prep_sel:
            PCA__n_components = st.sidebar.slider('PCAの次元数を選んでください。', min_value=1, max_value=X_len, value=X_len)
            pca_update = {'PCA': PCA(n_components = PCA__n_components)}
            prep_dict.update(pca_update)
        
        st.write('前処理を実行してください。')
        
        if st.sidebar.checkbox('前処理を実行'):

            # 選択された前処理リストをタプルのリストに変換
            for key in prep_dict.keys():
                if key in prep_sel:
                    prep.append((key, prep_dict[key]))

            #前処理
            pipe = Pipeline(prep)
            X = pipe.fit_transform(X)
            
            st.write('前処理の内容：')
            st.text(pipe)

            st.write('処理後のXの情報：')
            Df_info(pd.DataFrame(X))

            st.write('次の処理に進んでください。')

# 学習を実行

    st.sidebar.subheader('学習')    
    if st.sidebar.checkbox('学習を実行する'):

        st.header('学習')
        st.write('左のメニューから、処理を選択してください。')

        if not flg_impute and not flg_prep:
            X = train_df.drop(drop_columns, axis=1)
            y = train_df[y_column].astype(str)

        # ダミー化

        if not flg_prep and len(X.select_dtypes(include=object)) > 0:
            X = pd.get_dummies(X)

        # モデルを選択
        
        model_type = st.sidebar.radio('分類(Classifier) or 回帰(Regressor)', ['Classifier', 'Regressor'])
        clf_dict = {'Random Forest Classifier': RandomForestClassifier()}
        rgr_dict = {'Random Forest Regressor': RandomForestRegressor()}
        if model_type == 'Classifier':
            model_sel = st.sidebar.selectbox('Select Moodel', list(clf_dict.keys()))
            model = clf_dict[model_sel]
            scoring = st.sidebar.selectbox('Scoring', ['accuracy', 'f1_weighted', 'roc_auc_ovr_weighted'], index=0)

        if model_type == 'Regressor':
            model_sel = st.sidebar.selectbox('Select Moodel', list(rgr_dict.keys()))
            model = rgr_dict[model_sel]
            scoring = st.sidebar.selectbox('Scoring', ['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'], index=0)
        
        cv = st.sidebar.slider('Cross Validation', min_value=2, max_value=10, value=5)

        if st.sidebar.checkbox('学習を実行'):
            # Cross Validationを実施
            cv_scores = cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=-1, return_train_score=True)
            train_score = np.mean(cv_scores['train_score'])
            test_score = np.mean(cv_scores['test_score'])

            model.fit(X, y)

            flg_fit = True

            st.write('学習が完了しました。')
            st.write('Train Score :', train_score)
            st.write('Test Score :', test_score)
            st.write('次の処理に進んでください。')

# 学習結果の保存

            result_name = st.sidebar.text_input('学習ケース名', value='最適化前')
            result_memo = st.sidebar.text_input('学習ケースのメモ')

            # 結果を保存するDataframeの準備
            if flg_fit and st.sidebar.checkbox('学習結果の保存'):
                result_df.loc[result_name] = [train_score, test_score, result_memo] 

                st.write('学習結果の比較')
                st.write(result_df)

                # CSVの保存
                tmp_download_link = download_link('pd', result_df, 'result_df.csv', '学習結果のCSVファイルをダウンロード')
                st.markdown(tmp_download_link, unsafe_allow_html=True)

                # Pipelineの保存
                if flg_prep:
                    tmp_download_link = download_link('pkl', pipe, 'pipe.pickle', 'PipelineのPickleをダウンロード')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)

                # モデルの保存
                tmp_download_link = download_link('pkl', model, 'model.pickle', 'ModelのPickleをダウンロード')
                st.markdown(tmp_download_link, unsafe_allow_html=True)

# Randomized Search

    st.sidebar.subheader('パラメーター最適化')    
    if st.sidebar.checkbox('パラメーターを最適化する'):

        st.header('パラメーター最適化')
        st.write('左のメニューから、処理を選択してください。')
        if model_sel == 'Random Forest Classifier':
            RF__criterion = st.sidebar.multiselect('criterion', ['gini', 'entropy'], default=['gini', 'entropy'])
        if model_sel == 'Random Forest Regressor':
            RF__criterion = st.sidebar.multiselect('criterion', ['mae', 'mse'], default=['mae', 'mse'])
        if model_sel in ['Random Forest Classifier', 'Random Forest Regressor']:
            RF__n_estimators = st.sidebar.slider('n_estimators', min_value=50, max_value=1000, value=(50, 200), step=50)
            RF__max_depth = st.sidebar.slider('max_depth', min_value=1, max_value=100, value=(10, 50))
            RF__min_samples_split = st.sidebar.slider('min_samples_split', min_value=2, max_value=10, value=(2, 5))
            RF__min_samples_leaf = st.sidebar.slider('min_samples_leaf', min_value=1, max_value=10, value=(1, 5))
        n_iter = st.sidebar.slider('Number of Iteration', min_value=2, max_value=500, value=100)

        if st.sidebar.checkbox('最適化の実行'):

            if model_sel in ['Random Forest Classifier', 'Random Forest Regressor']:
                rscv_params = {
                    'criterion': RF__criterion,
                    'n_estimators': RF__n_estimators,
                    'max_depth': RF__max_depth,
                    'min_samples_split': RF__min_samples_split,
                    'min_samples_leaf': RF__min_samples_leaf,
                }

            rscv = RandomizedSearchCV(model, rscv_params, n_iter=n_iter, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
            start = time.time()
            rscv.fit(X, y)
            search_time = time.time()-start
            st.write('最適化が完了しました。({:.1f} seconds)'.format(search_time))
            st.write('Best Score :', rscv.best_score_)
            st.write('Best Parameter :', rscv.best_params_)
            cv_results_df = pd.DataFrame(rscv.cv_results_)[['params', 'mean_train_score', 'mean_test_score', 'rank_test_score']]
            st.dataframe(cv_results_df.style.bar(width=90, color='pink'))
            model = rscv.best_estimator_

# 学習結果の保存

            result_name = st.sidebar.text_input('学習ケース名(2)', value='最適化後')
            result_memo = st.sidebar.text_input('学習ケースのメモ(2)')

            # 結果を保存するDataframeの準備
            if st.sidebar.checkbox('学習結果の保存(2)'):
                result_df.loc[result_name] = [train_score, test_score, result_memo]

                st.write('学習結果の比較')
                st.write(result_df)

                # CSVの保存
                tmp_download_link = download_link('pd', result_df, 'result_df.csv', '学習結果のCSVファイルをダウンロード')
                st.markdown(tmp_download_link, unsafe_allow_html=True)

                # モデルの保存
                tmp_download_link = download_link('pkl', model, 'model.pickle', 'ModelのPickleをダウンロード')
                st.markdown(tmp_download_link, unsafe_allow_html=True)

# 予測データの読み込み

if flg_fit:
    st.header('予測データの読み込み')
    st.write('左のメニューから、予測するデータを選んでください。')

    st.sidebar.subheader('予測データの読み込み')
    pred_file = st.sidebar.file_uploader("予測ファイルのアップロード", type='csv')

    if pred_file is not None:

        # アップロードファイルをメイン画面にデータ表示
        pred_df = Load_data(pred_file)

        if y_column in drop_columns:
            drop_columns.remove(y_column)
        if drop_columns:
            X_pred = pred_df.drop(drop_columns, axis=1)
        else:
            X_pred = pred_df

# 予測データの欠損値の処理

        if flg_impute:
            if impute_drop:
                X_pred.dropna(axis=0, subset=impute_drop, inplace=True)
            if impute_mean:
                X_pred[impute_mean] = pipe_mean.fit_transform(X_pred[impute_mean])
            if impute_med:
                X_pred[impute_med] = pipe_med.fit_transform(X_pred[impute_med])
            if impute_freq:
                X_pred[impute_freq] = pipe_freq.fit_transform(X_pred[impute_freq])
            st.write('欠損値の処理後のXの情報：')
            Df_info(X_pred)

        if X_pred.isnull().values.sum() > 0:
            st.write('欠損値があります。学習データの欠損値の処理まで戻ってやり直してください。')

# 予測データの前処理

        if flg_prep:
            if len(X_pred.select_dtypes(include=object)) > 0:
                X_pred = pd.get_dummies(X_pred)
            X_pred = pipe.fit_transform(X_pred)

            st.write('前処理後のXの情報：')
            Df_info(pd.DataFrame(X_pred))

# ダミー化
        if not flg_prep and len(X_pred.select_dtypes(include=object)) > 0:
            X_pred = pd.get_dummies(X_pred)

# 予測
        st.write('左のメニューから、予測を実行してください。')

        st.sidebar.subheader('予測')
        if st.sidebar.checkbox('予測を実行する'):
            st.header('予測')
            y_pred = model.predict(X_pred)
            pred_df[y_column] = y_pred

            st.write('予測したファイル：')
            st.write(pred_df)
            st.write('{}行のデータを予測しました。'.format(pred_df.shape[0]))

            # CSVの保存
            tmp_download_link = download_link('pd', pred_df, 'predicted.csv', '予測結果のCSVファイルをダウンロード')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
