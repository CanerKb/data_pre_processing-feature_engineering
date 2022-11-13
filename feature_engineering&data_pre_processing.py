#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

# feature : Degisken / Ozellik
# agaca dayali yontemlerde aykiri degerlerin etkisi gorece daha dusuktur.
# Robast Dagilim Olcusu: degerlerin bir cogunun o aralikta dagilmasidir.
    # IQR: Q3 - Q1 robast dagilimdir. Degerlerin bir cogu 25'lik ile 75'lik degerler arasindadir.

# gerekli kutuphanelerin import edilmesi
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

# Bazi Gorsel Ayarlar
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# Veri setini iceri aktarmak icin fonksiyon tanimlamak
def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data

df = load_application_train()
df.head()


def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data


df = load()
df.head()



#############################################
# 1. Outliers (Aykırı Değerler)
#############################################

#############################################
# Aykırı Değerleri Yakalama
#############################################

###################
# Grafik Teknikle Aykırı Değerler
###################

sns.boxplot(x=df["Age"])
plt.show(block=True)

###################
# Aykırı Değerler Nasıl Yakalanır?
###################

q1 = df['Age'].quantile(0.25)
q3 = df['Age'].quantile(0.75)
iqr = q3 -q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

df[(df['Age'] < low) | (df['Age'] > up)]

# Aykiri deger index bilgisi indekslere ulasmak.
df[(df['Age'] < low) | (df['Age'] > up)].index

# fonksiyonlastirmak icin True/False degerleriden gore islem yap diyebilirim.
# ya da dongude sorgulamak icin.
df[(df['Age'] < low) | (df['Age'] > up)].any(axis=None)
# Out[21]: True
df[(df['Age'] < low)].any(axis=None)
# Out[22]: False

# 1. Eşik değer belirledik.
# 2. Aykırılara eriştik.
# 3. Hızlıca aykırı değer var mı yok diye sorduk.

###################
# İşlemleri Fonksiyonlaştırmak
###################

# islemleri fonksiyonlastirirken asama asama fonksiyonlastirmamiz lazim.
# oncelikle bir threshold fonskiyonu tanimlamak Sonrasinda Aykirilik sorgusu yapan bir fonk. tanimlamak gerekir.

"""
Denemeler: 

def outlier_thresholds(dataframe, col_name, q1= 0.25, q3= 0.75):
    quartile1= dataframe[col_name].quantile(q1)
    quartile3= dataframe[col_name].quantile(q3)
    IQR= quartile3 - quartile1
    up_limit= quartile3 + 1.5 * IQR
    low_limit= quartile1 - 1.5 * IQR
    return  low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis= None):
        return True
    else:
        return False

outlier_thresholds(df, 'Age')
check_outlier(df, 'Age')
"""

# esik degerleri yakalamak
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "Age")
outlier_thresholds(df, "Fare")

low, up = outlier_thresholds(df, "Fare")

# fare degiskeninin ayikiri olan ilk 5 satirini gozlemlemek
df[(df["Fare"] < low) | (df["Fare"] > up)].head()

# fare degiskeninin ayikiri olan degerlerinin indez bilgisine erismek
df[(df["Fare"] < low) | (df["Fare"] > up)].index

# aykiri deger var midir!?
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age")
check_outlier(df, "Fare")

###################
# grab_col_names
###################

dff = load_application_train()
dff.head()

# degisken tiplerini yakalamak; numerik, kategorik ve kardinal olanlari.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    # cat_cols: df kolonlarinda gez tipi object olanlar kategoriktir.
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    # numerik fakat kategorik: df kolonlarinda gez essiz sinif sayisi cat_th den kucuk olanlar kategoriktir ve
    # ayni zamanda tipi object olmayanlar: zaten yukarda objectleri aldik burada amac sayisal siniflandirmayi yakalamak.
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    # categorik ama kardinal: df kolonlarinda gez essiz sinif sayisi car_th den buyuk olanlar kardinaldir bunu yaparken
    # ayni zamanda kolon dtype object olanlarda yap.
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    # cat_cols u guncellemek lazim: numerik gorunumlu ama kategorik olanlari cat_cols a ekleriz ve cardinal'leri
    # ondan cikaririz.
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    # num_cols: df kolonlarinda gez ve tipi object olmayanlari listele.
    # yalniz burada date/tarih'ler de gelecektir sonradan cikartabiliriz. Manuel olarak.
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
"""
cat_cols, num_cols, cat_but_car = grab_col_names(df)
Observations: 891


Variables: 12
cat_cols: 6
num_cols: 3
cat_but_car: 3
(12 = 6 + 3 + 3)

num_but_cat: 4
num_but_cat zaten cat_cols'un icinde yer alir ek bilgi olarak doner.
"""

# numeric kolonlar icerisine ID degiskeni de yer almakta onu cikarmamiz gerekmekte.
num_cols = [col for col in num_cols if col not in "PassengerId"]

# numaric kolonlar(Degiskenler) icerisinde aykirilik var midir check ediyoruz.
for col in num_cols:
    print(col, check_outlier(df, col))


cat_cols, num_cols, cat_but_car = grab_col_names(dff)

num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

# tekrar check ediyorum
for col in num_cols:
    print(col, check_outlier(dff, col))

###################
# Aykırı Değerlerin Kendilerine Erişmek
###################

# Ağaca dayalı algoritmalar aykırı ve eksık değerlere DUYARSIZdır Agaca dayalı yöntemler kullanıldıgında aykırı ve eksık
    # degerler göz ardı edilir edilmelidir de!

# Aykırılıkların index bilgisi ve 10 eşik degerınde aykırılıkları gozlemlemek.
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    # alt ve üst aykırı degerler 10 dan buyukse ilk 5 gözlemi görmek ıstıyorum.
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    # alt ve ust aykırı degerler 10dan kucukse direkt hepsini görmek ıstıyorum.
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
    # index argümanını True yaparsam index bilgilerini almak ıstıyorum dıyorum.
    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "Age")

grab_outliers(df, "Age", True)

age_index = grab_outliers(df, "Age", True)

# aykırı degerlerin alt ve ust limitleri nelerdır?
outlier_thresholds(df, "Age")
# aykırı deger var mı yok mu ?
check_outlier(df, "Age")
# varsa gözlemleyıp indeks bilgisini alalım.
grab_outliers(df, "Age", True)

#############################################
# Aykırı Değer Problemini Çözme

# Baskılama Yöntemi (re-assignment with thresholds)
###################

low, up = outlier_thresholds(df, "Fare")

df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"] # kosul sonrasında ındexleme ıslemı yaptım.
# bir ust satir ile ayni ciktiyi alirim.
df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"] # loc ile seçtım o kolonu

df.loc[(df["Fare"] > up), "Fare"] = up

df.loc[(df["Fare"] < low), "Fare"] = low

# fonksıyon ıle baskılama ıslemı yapıyorum.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Titanic veri setini yukluyorum.
df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

#############################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
#############################################

                                # Yorumsal notlar....

# 17, 3
# tek basına anlamlı olan yapılar bırden fazla degıskenle bırlıkte anlamsızlıga burunurler.
# 17 yaşında olup 3 kere evlenmek gıbı bır ornek buna ornek olacaktır. Anormal durumdur aykırılık oluşur.


# Lof yöntemı uzaklık ve yogunluk tabanlı çalışır. komşuluk sayısı belırtırsın ve en uzak komşusu kadar yarıçap çizerek
    # ıcınde kalan yogunluğu degerlendırır. bunu her bır verı ıcın yapar.
# lof yontemınde bır skor alırız ve bu skor 1'e ne kadar yakınsa o kadar ıyıdır.
    # çok degıskenlı verı setlerını ıkı boyutluda gozlemleyebılmek şu şekilde mumkun olacaktır:
        # PCA ile toplam degısken sayısını en anlamlı ıfade edebılecek 2 degıskene ındırırız ve 2D grafık halınde yapablırız.
    # ıkı boyutlu gorselleştırme de lof ta eşik deger olarak mısal 5 sectık grafıkte 5 ten buyuk olanlar aykırı degerdır.

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()
df.shape
for col in df.columns:
    print(col, check_outlier(df, col))


low, up = outlier_thresholds(df, "carat")

df[((df["carat"] < low) | (df["carat"] > up))].shape
# Out[52]: (1889, 7)
# tek değişkenli baktıgımızda 1889 adet aykırı gozlem var tabı burada q1:0.25 ve q3:0.75 iken. bunu azaltabılırız.

low, up = outlier_thresholds(df, "depth")

df[((df["depth"] < low) | (df["depth"] > up))].shape
# Out[54]: (2545, 7) tek değişkenlı baktıgımızda durum bu.

# çok değişkenlı bakış ıle:

# burada local outlıer fonks. on tanımlı degerı 20 ı getırdık cunku en ıyısını bulmayı yorumlamak çok çok guç ve
    # muhtemelen yapılamaz. sadece gorseleştırıp ona gore bır yorum v ekırılım degerlendırırlır.


clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

# clf'den skor üretiyorsun
df_scores = clf.negative_outlier_factor_
# üretilen bu skoru gözlemliyorsun 1 en yakın en iyi buradaki bir -1(negatıf 1)'dir.
# Skorlara bakıp anı kırılmalar gözlemlemek zordur o yüzden bu skorları DataFrame çevırıp grafik ile yorumlamak gerek.
df_scores[0:5]
# df_scores = -df_scores
# skorları sıralayıp yorumlamak. İndex sıfırdan başlıyor hatırlatma.
np.sort(df_scores)[0:5]


# sıraladığımız skorları DataFrame cevırıp score adlı değişkene atadık
scores = pd.DataFrame(np.sort(df_scores))
# scores adlı DataFrame grafiğini çiziyoruz. Burada xlim ile oynayıp en ıyı kırılmaya karar verebılırız.
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show(block=True)
# grafık gözlemleneceği üzere 3. index ten sonra anı kırılımlar kaybolmakta.

# eşik deger olarak 3. ındex ı seçiyoruz.
th = np.sort(df_scores)[3]

df[df_scores < th]

df[df_scores < th].shape


df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)


#############################################
# Missing Values (Eksik Değerler)
#############################################

#############################################
# Eksik Değerlerin Yakalanması
#############################################

"""
# eksik verinin rassallıgı çok önemli. Eksik verı rastgele mı ortaya cıktı yoka yapısal prroblemlerden mı ona göre eksik
    # verı doldurma işlermleri yapılmalıdır.
# Nitekim:
    # Eksik gözlemlerin veri setinden direkt çıkarılabilinmeleri için veri setindeki eksikliğin bazı durumlarda kısmen
        # bazı durumlarda tamamen rastlantısal olarak oluşmuş olması gerekmektedir.
    # Eğer eksiklikler değişkenler ile ilişkili olarak ortaya çıkan yapısal problemler ile meydana gelmiş ise bu durumda
        # yapılacak silme işlemleri ciddi yanlılıklara sebep olabilecektir.
                                                                    # Tabachnick ve Fidell , 1996

# Örneğin kredi kartı kullanmayan kişilerin kredi kartı harcamalarının SIFIR olması bu rassal degıldır. Bu durumunda 
    # yapılacak sılme işlemi ciddi yanlılıklara sebep olacaktır. Onun yerine uygun* doldurma yöntemlerı seçilmelidir. 
    
"""


df = load()
df.head()

# eksik gozlem var mı yok mu sorgusu True/False
df.isnull().values.any() # df.isnull() eksik deger varmı .values'te bu durumu tutarz.herhangi birinde(hiç) diye sorarız.
# df.isnull() dataframe uzerınde gösteririr.
# df.isnull().values array olarak degerlerde gösterir.
# df.isnull().values.any() 1 tane bile Na NaN varsa True döndürür.


# degiskenlerdeki eksik deger sayisi
df.isnull().sum()

# degiskenlerdeki tam deger sayisi
df.notnull().sum()

# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum()

# en az bir tane eksik degere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

# Azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

# eksik degerlerin yüzdece oranı nedir?

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

df.shape
# eksik değer barındıran kolonlar hangileridir.
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

# tam deger barındıran kolonlar hangileridir.
full_cols = [col for col in df.columns if df[col].notnull().sum() == df.shape[0]]

# missing_values_table ile null olan değişkenlerin isimlerini frekanslarını ve oranları görüntüleyebiliyoruz.
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

#############################################
# Eksik Değer Problemini Çözme
#############################################

missing_values_table(df)

#############################################
# Çözüm : Tahmine Dayalı Atama ile Doldurma
#############################################

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

# encoder - Dönüşüm işlemleri yapmamız lazım. Algoritmalar için. Binary ifade etmek lazı kategorik değişkenleri.
# kategoriklerle numerikleri bır arada verdık ama zaten get_dummies metodu sadece kategorıklere yapar bu işleri
# Ancak SibSp zaten numeric halde olan bir kategorık değişken oldugu ıcın ona şimdilik dokunmadık
# df içerisindeki cardinal değişkenler harıcınde olan numeric ve kategorık degıskenleri dff içerisinde OHE ile atıyorum.
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

dff.head()

# değişkenlerin standartlatırılması  bu da bır ıhtıyactır. Makıne öğrenmesi tekniğini kullanmak ıcın.
# ölçeklemeyi(MİNMAXSCALER()) scaler a atıyorum
scaler = MinMaxScaler()
"""
Ölçekleme işlemi sonrası alacagımız çıktı vektördür. bu yüzden dataframe cevirdik

Önemli Not: her estimatör sonrası yapılacak estimatör işlemi bir vektördür

"""
# scaler.fit_transform(dff) array formunda gelecegı için ölçekleme işlemi
# vektörümüzü DataFrame ceviriyoruz.
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
# böylelikle tekrar DataFrame e cevirdik.
dff.head()


# knn'in uygulanması.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5) # estimatör oluşturuldu
# estimatör uygulandıgında fit_transform(dff) ile çıktı vektör oldugu ıcın  dataframe cevirdik.
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
# burada tahmin ile doldurma işlemi gerçekleşti peki hangi NaN degerleri doldurdu ne doldurdu gözlemlemek ıstesem !
# ve Burada head() de de görecegımız uzere scale edlmiş haldeler 0-1 arasındalar(defoult). bunu verisetinin ılk
# halındekıne cevırmak gerekecek
dff.head()
# scale leme işlemini gerıye almak için .inverse_transform(dff)
# burada yine estimatörü .inverse_transform(dff) ile uyguladıgımız ıcın vektor elde edilecek bunu dataframe ceviriyoruz.
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()

# asıl veri setimiz olan df içeisine dff^deki yenı değerleri atayalım
df["age_imputed_knn"] = dff[["Age"]]
# df içerisinde yukardakı kodtakı iki kolonda var artık. bunları yan yana loc ile gözlemleyelim.
df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
df.loc[df["Age"].isnull()]


#############################################
# Gelişmiş Analizler
#############################################

###################
# Eksik Veri Yapısının İncelenmesi
###################

# bar ile eksik shape teki gözlem adetini görebiliyorsun. grafiksel oalrak
msno.bar(df)
plt.show(block= True)

# burada yapısal ya da rassallığı gözlemliyoruz. ÖNEMLİ
msno.matrix(df)
plt.show(block= True)


# eksik degerlerin birlikte oluşup oluşmamasını incelemiş oluyoruz. Eksiklik Korelasyonu.
# Pozitif ise grinde eksiklikvarsa dıerınde de vardır.
# negatif ise biri eksık mise dıgerı doludur.
# bunlar hep bagımlılık yapısını dusundurmelidir.
# korelasyonlar %60'ın uzerınde çıkıyorsa kayde deger olabılırdı.
msno.heatmap(df)
plt.show(block= True)

###################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
###################
# Eksikliklerin bagımlı degişken ile ilişkisi her zaman önemlidir.



missing_values_table(df, True)
na_cols = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Survived", na_cols)
#                TARGET_MEAN  Count
# Cabin_NA_FLAG
# 0                    0.667    204
# 1                    0.300    687

# Cabin değişkenınde çok fazla sayıda boşluk vardı  yukarıda bu kabin numarası olmayanların hayatta kalma ortalaması
    # 0.30 imiş ve biz bunları kabin numarası olmayanlar dıye ayırabılırız gibi.

#############################################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#############################################

#############################################
# Label Encoding & Binary Encoding
#############################################
# label encoding alfabetık sıraya gore 0,1,2,3.... değerlerini verır.
df = load()
df.head()
df["Sex"].head()

le = LabelEncoder() # estimatör oluştu. her estimator fit_trasform edildiginde vektor verır.
le.fit_transform(df["Sex"])[0:5]
# le içerisinde dönüşmüş ve donusmemıs bılgıler saklıdır. ve inverse_transform ile ([0,1]) hangıleriydi diye öğrenırım.
le.inverse_transform([0, 1])

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

df = load()


# 2li encod işleminde label_encoder seçeceksek buyuk verı setlerinde bu durumu yakalayabılmek ıcın aşağıda
    # list compresion yapısından faydalana biliriz.
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

# burada df[col].nunique() yerine len(df[col].unıque()) == 2 yaparsak unıque() fonksiyonu NaN larıda bır sınıf olarak
    # göreceği için misal Sex değişkenınde NaN olma durumunda bu boş hucreyıde bır sınıf olarak gorecektı. len() sonucu
    # 3 çıkacaktı.


for col in binary_cols:
    label_encoder(df, col)

df.head()

df = load_application_train()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

df[binary_cols].head()

#  NaN değerlere de label_encoder 2 değerini atadı. normalde eksık degerleri değişken dönüşümü öncesinde ele alırız
    #  ola ki almadık. Bilmeliyiz ki 2 ataması NaN 'ları ifade etmektedir.



for col in binary_cols:
    label_encoder(df, col)


df = load()
df["Embarked"].value_counts()
# Out[38]:
# S    644
# C    168
# Q     77

df["Embarked"].nunique()
# Out[39]: 3

len(df["Embarked"].unique())
# len(df["Embarked"].unique())
# Out[40]: 4
df["Embarked"].unique()
# Out[41]: array(['S', 'C', 'Q', nan], dtype=object)

#############################################
# One-Hot Encoding
#############################################

df = load()
df.head()
df["Embarked"].value_counts()

pd.get_dummies(df, columns=["Embarked"]).head()

pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()

# Boş olan hücrelere-degerlere NaN lere de bir değer ataması yapmak için get_dummies içerisinden
    # dummy_na argümanını=True yaparız
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()

pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = load()

# cat_cols, num_cols, cat_but_car = grab_col_names(df)

# burada OHE yapacagım değişkenleri kendim seçiyorum
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]


one_hot_encoder(df, ohe_cols).head()

# değişikliklerin kalıcı olmasını istiyorsam atama yapmalıyım.
# df= one_hot_encoder(df, ohe_cols).head()


df.head()

#############################################
# Rare Encoding
#############################################

# Rare -Nadir Demektir.

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.

###################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
###################

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()
# çıktı da en az ki olan
# Academic degree                     164

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col)

###################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
###################

df["NAME_INCOME_TYPE"].value_counts()
# Out[55]:
# Working                 158774
# Commercial associate     71617
# Pensioner                55362
# State servant            21703
# Unemployed                  22
# Student                     18
# Businessman                 10
# Maternity leave              5
# Name: NAME_INCOME_TYPE, dtype: int64

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()
# Out[56]:
# NAME_INCOME_TYPE
# Businessman            0.000
# Commercial associate   0.075
# Maternity leave        0.400
# Pensioner              0.054
# State servant          0.058
# Student                0.000
# Unemployed             0.364
# Working                0.096
# Name: TARGET, dtype: float64
#
# Yukardaki df["NAME_INCOME_TYPE"] ı hedef değişkene göre mean degerlerinin analizini yaptıgımızda Out[55] ile
    # kıyaslamalıyız çünkü şöyle out[55] de son dört değişkenın sınıf frekansı çok az bunları Rare olarak mı
    # birleştirsem diyorsun ancak target ı etkileme ortalamaları aynı degıl ama yıne rare olarak bırleştırebılırsın
    # yorum sana kalmış. Misal businessman ın etkisinin ortalaması ile Unemployed'ın bambaşka. eger yakın olsalardı
    # rahatca yapdık. Eklersek Rare olarak bır gurultumu ekledık mi acaba dıye dusunmelıyız de.


# rare_analyser fonksiyonu ile categorık değikenlerde rare analızı yaparız.
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)

#############################################
# 3. Rare encoder'ın yazılması.
#############################################

# önce ki bölümde degerlendirme yaptık şimdi de analiz sonrası yazma işlemindeyiz.


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "TARGET", cat_cols)

df["OCCUPATION_TYPE"].value_counts()


#############################################
# Korelasyon analizi
#############################################


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [upper_triangle_matrix.loc[(upper_triangle_matrix[col] > corr_th) , col] for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show(block= True)
    return drop_list

drop_list = high_correlated_cols(df, plot=True, corr_th=0.92)


#############################################
# Feature Scaling (Özellik Ölçeklendirme)

# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
###################

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

df.head()

age_cols = [col for col in df.columns if "Age" in col]

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in age_cols:
    num_summary(df, col, plot=True)

###################
# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
# Binning
###################

df["Age_qcut"] = pd.qcut(df['Age'], 5)

#############################################
# Feature Extraction (Özellik Çıkarımı)
#############################################

#############################################
# Binary Features: Flag, Bool, True-False
#############################################

df = load()
df.head()

#Cabin degiskenine bakiyorum
df.groupby('Survived').agg({'Cabin' : 'count'})
# cabin degerlerinde bos olanlara 0, dolu olanlara 1 ekleyelim.
# kabini var olanlar ile olmayanlari kiyaslamis olmak icin degisken turetiyoruz.
df["NEW_CABIN_BOOL"] = df['Cabin'].notnull().astype('int')

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

df.groupby("NEW_CABIN_BOOL").agg({"Survived": 'mean')
df.groupby("Survived").agg({"NEW_CABIN_BOOL": "count"})


from statsmodels.stats.proportion import proportions_ztest

# proportions_ztest testi sunu ders iki degisken arasinda anlamli bir fark yoktur der. p<0.05 ise RED!

# AB Test uygulamak
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# bu degiskeni veri setine degisken olarak eklemek icin bu test yeterli olacaktir. sonrasinda makine ogrenmesi modelimiz
# oncesi  onemli degiskenlere bakacagiz simdilik bunu oraya yollayabilir olacagiz cunku istatistiksel olarak anlamli bir
# fark vardir cikti p <0.05. Yine de dogrudan bu degisken hayatta kalma icin birebir etki yaratiyor kanisina varamayiz ama
# bir etkisi oldugunu yabana atamayiz. o yuzden degisken olarak yanimizda goturuyoruz.
# Tek basina anlamli ama coklu etki gozonune makine ogrenmesinde degerlendirilecek.


# bir kisi yalniz mi degil mi belki hayatta kalma durumunu tetiklemistir.
df.head()
df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"
df.head()
df.groupby("NEW_IS_ALONE").agg({"Survived": ["mean",'count']})


test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Cikti: Test Stat = 9.4597, p-value = 0.0000 p<0.05 oldugu icin H0 RED! o zaman deriz ki anlamli fark vardir.
# df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum() survived durumunda baktigimiz icin hayatta kalanlari 1leri sum()
    # edecek. Yani yalniz olmayan ve hayatta kalan kac kisi var. basari sayisi
# df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0] gozlem sayisini verecek.


#############################################
# Text'ler Üzerinden Özellik Türetmek
#############################################
df = load()
df.head()


###################
# Letter Count  - isimlerdeki kelimeleri sayalim.
###################
df['NEW_NAME_COUNT'] = df['Name'].str.len()  # bosluklari ve diger ozel karakterleri de saydik
df.head()

###################
# Word Count
###################
df['NEW_NAME_WORD_COUNT'] = df['Name'].apply(lambda x: len(str(x).split(" ")))
df.head()
# asagidaki kodda olurdu zaten Name degerleri str'dir.
df['NEW_NAME_WORD_COUNT2222'] = df['Name'].apply(lambda x: len(x.split(" ")))


df['Name'].str.split()[:5]

# df['Name'].split()[:5] hata verir! Series'tir, str degildir.

###################
# Özel Yapıları Yakalamak
###################

# burada dr olanlarin hayatta kalma durumlari ilginc buldum. Bakalim dr olmak nasil etkili.

df["NEW_NAME_DR"] = df['Name'].apply(lambda x : len([x for x in x.split() if x.startswith('Dr')]))

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df.head()
df.groupby('NEW_NAME_DR').agg({'Survived' : ['mean','count']})


test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_NAME_DR"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_NAME_DR"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_NAME_DR"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_NAME_DR"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.7596, p-value = 0.4475  p-value < 0.05 old. icin anlamli bir durum oldugu dusunulebilir ancak 0.05 e
# de oldukca yakindir. ve count degeri dr'lerin 10 oldugu icin bu degiskeni turetmekten vazgecebilriiz. bunlar yorum.

###################
# Regex ile Değişken Türetmek
###################
# metinlerin uzerinde calismaya devam ediorum title lari cekecegiz.


df.head()
# mesleklerine gore degisken olusturmak.
df['NEW_TITLE111111'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
# \.  kelimenin sonu nokta ile bitsin
# A-Za-z buyuk ya da kucuk harflerden olusacak sekilde karekterleri yakala diyorum.

# deneme
df['NEW_TITLE222222'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=True)


df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})
# BURADA MESELEK GRUPLARINA GORE BOS HUCRELERI ATAYABILECEGIMIZ GIBI FREKANSI KUCUK OLANLARI RARE ILE BIR DEGISKENE
    # CEVIREBILIRIZ.

#############################################
# Date Değişkenleri Üretmek  - tarih degiskenleri ile degisken uretmek
#############################################

dff = pd.read_csv("datasets/course_reviews.csv")
dff.head()
dff.info()
# 1 Timestamp object
# 2 Enrolled  object

# Bunlarin tipini oncelikle datetime a cevirmek gerekir.

dff["Timestamp"] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")
# dff['Timestamp'] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")
dff.head()
dff.info()
# sunu yaptik Timestamp 'in tipini degistirdik datetime64[ns] olarak.
# simdi tek tek yillari aylari ve gunleri cekelim

"""
https://docs.rapids.ai/api/cudf/stable/api_docs/api/cudf.core.series.DatetimeProperties.is_quarter_start.html

DatetimeProperties icerisindeki fonksiyonlar. 



"""

# year
# dff["year"] = dff["Timestamp"].dt.year
dff['year'] = dff['Timestamp'].dt.year

# month
# dff["month"] = dff['Timestamp'].dt.month
dff['month'] = dff['Timestamp'].dt.month

# day
dff["days"] = dff['Timestamp'].dt.day

# day name - haftanin gunleri
dff['day_name'] = dff['Timestamp'].dt.day_name()

# hour - saat i yakalamak
dff["hour"] = dff["Timestamp"].dt.hour

# minute - dakika yi almak.
dff["minute"] = dff["Timestamp"].dt.minute

# second - saniyeleri yakalamak
dff["second"] = dff["Timestamp"].dt.second

# # weekday - haftanin kacinci gunu
# dff["day_of_week"] + dff["Timestamp"].dt.days_of_week
# dayofweek bunlar hata verdi.

# # days_in_week - haftanin hangi gunu oldugu yakalamak
# dff["days_in_week"] = dff["Timestamp"].dt.days_in_week
# bulunamadi olamadi.

# dff["day_in_week"] = dff["Timestamp"].dt.days_in_weeks

# day_of_year - yilin hangi gunleri oldugunu yakalamak
dff["day_of_year"] = dff["Timestamp"].dt.day_of_year

# days_in_month - ayin hangi gunu oldugunu yakalamak
dff["days_in_month"] = dff["Timestamp"].dt.days_in_month


# simdinin tarihi
date.today().year

# year diff   : simdinin tarihinden veri setinin yillarini cikarmak istersem.
# Yani kac sene onceye dayaniyor bilmek istersem.
dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year


# month diff (iki tarih arasındaki ay farkı): yıl farkı + ay farkı
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month
# oncelikle iki tarih arasindaki farki almaam gerekir ve sonrasinda bu farki 12 ile carpmaliyim. bu yillarin ay farki
    # olur. sonrasinda uzerine aylari eklemem gerekir.
# simdiden kac ay oncesine dayanmakta. bunu bilmek istesem

# dff["month_diff"] = (date.today().year - dff["Timestamp"].dt.year)*12 + (date.today().month - dff["Timestamp"].dt.month)


dff.head()

# date


#############################################
# Feature Interactions (Özellik Etkileşimleri)
#############################################

# degiskenlerle etkilesim kurmaktir birbiriyle carpmak bolmek vs
df = load()
df.head()

# refah seviyesi durumu olusturmaya calisabiliriz.
# soyle ki yasai kucuk olup gemi de yer bulabiliyor olmak bir gosterge olabilir mi ya da tam tersi!
df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1 # +1 de kendisi

# loc 'un bambaska kullanimi: hem degisken uretmek hem de uretilen degisken icerisine diger degiskenlerden turetilen
    # kosullar ile atamalar yapmak.


# cinsiyetleri yaslarina gore kategorize edebilirim. genc olanlar, olgun olanlar, daha olgun olanlar.
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'

df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'

df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'

df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df.head()

df.groupby("NEW_SEX_CAT")["Survived"].mean()

len(df["NEW_AGE_PCLASS"])
df["NEW_AGE_PCLASS_CUT"] = pd.qcut(x=df["NEW_AGE_PCLASS"], q=4, labels=['D','C','B','A'] )
df.groupby("NEW_AGE_PCLASS_CUT").agg({"Survived" : ["mean", "count"]})
