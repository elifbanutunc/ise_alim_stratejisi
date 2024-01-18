# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:37:18 2024

@author: TUNC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# CSV dosyasını oku (dosya adını ve yolunu güncelleyin)
veri_seti = pd.read_csv('Job_Placement_Data.csv', delimiter=';')


# Veri setini incele
print(veri_seti)


veriler= {
    'name': ['John', 'Jane', 'Alice', 'Bob', 'Eva'],
   
    'hsc_percentage': [78, 85, 88, 82, 90],
    'undergrad_percentage': [85, 90, 92, 88, 95]
}

df = pd.DataFrame(veri_seti)

# Çubuk Grafiği
plt.figure(figsize=(10, 6))

plt.bar(df['name'], df['undergrad_percentage'], label='Undergrad Percentage', alpha=0.5)
plt.bar(df['name'], df['mba_percent'], label='SSC Percentage')
plt.xlabel('Öğrenci Adı')
plt.ylabel('Yüzde')
plt.title('UNDERGRAD ve MBA Yüzdelikleri')
plt.legend()
plt.show()


# Örnek veri seti (örneğin bir CSV dosyasından okunabilir)
veripasta = {
'name': ['John', 'Jane', 'Alice', 'Bob', 'Eva','Ryan','Nora','William','Scarlet','John','Hazel','Jonathan'],
'program_used': ['Python', 'Java', 'C++',  'Java', 'Docker',' HTML/CSS', 'Objective-C', 'Delphi', 'Swift', 'F#','Git','Node.js']

}

df = pd.DataFrame(veripasta)

# 'program_used' sütunundaki değerleri say
program_kullanim_sayisi = df['program_used'].value_counts()

# Pasta Grafiği
plt.figure(figsize=(10, 6))
plt.pie(program_kullanim_sayisi, labels=program_kullanim_sayisi.index, autopct='%1.1f%%', startangle=140)
plt.title('Program Kullanımı Dağılımı')
plt.show()

bolum_pasta = {
    'name': ['John', 'Jane', 'Alice', 'Bob', 'Eva'],
    'undergrad_degree': ['Comm&Mgmt', 'Sci&Tech', 'Others', 'Comm&Mgmt', 'Sci&Tech']
}

df = pd.DataFrame(bolum_pasta)

# 'undergrad_degree' sütunundaki değerleri say
derece_sayisi = df['undergrad_degree'].value_counts()

# Pasta Grafiği
plt.figure(figsize=(8, 8))
plt.pie(derece_sayisi, labels=derece_sayisi.index, autopct='%1.1f%%', startangle=140)
plt.title('Lisans Bölümleri Dağılımı')
plt.show()




class Aday:
    def __init__(self, id_no, name, surname, science_exam_score, work_experience):
        self.id_no = id_no
        self.name = name
        self.surname = surname
        self.science_exam_score = science_exam_score
        self.work_experience = work_experience

    def is_alinabilir_mi(self):
        if self.science_exam_score > 80 and self.work_experience == "Yes":
            return 'İyi performans gösteren aday, işe alınabilir.'
        else:
            return 'Daha fazla değerlendirme yapılması gereken aday.'

# CSV dosyasını oku (dosya adını ve yolunu güncelleyin)
veri_seti = pd.read_csv('Job_Placement_Data.csv', delimiter=';')

# Aday sınıfını kullanarak veri setini oluştur
adaylar = []
for _, row in veri_seti.iterrows():
    aday = Aday(row['id_no'], row['name'], row['surname'], row['science_exam_score'], row['work_experience'])
    adaylar.append(aday)

# İşe alım stratejisi fonksiyonunu Aday sınıfına ekleyin
print("İşe alınacak adaylar:")
for aday in adaylar:
    sonuc = aday.is_alinabilir_mi()
    if sonuc == 'İyi performans gösteren aday, işe alınabilir.':
        print(f"ID: {aday.id_no}, {aday.name}, {aday.surname}")
# İşe alım stratejisi fonksiyonu
def is_alim_stratejisi(aday):
    if aday['science_exam_score'] > 80  and aday['work_experience']=="Yes":
        return 'İyi performans gösteren aday, işe alınabilir.'
    else:
        return 'Daha fazla değerlendirme yapılması gereken aday.'

#adaylar için işe alım stratejisini uygula
for _, aday in veri_seti.iterrows():
    sonuc = is_alim_stratejisi(aday)
    print(f"{aday['id_no']} {aday['name']} {aday['surname']}: {sonuc}")

print("İşe alınacak adaylar:")
for index, aday in veri_seti.iterrows():
    sonuc = is_alim_stratejisi(aday)
    if sonuc == 'İyi performans gösteren aday, işe alınabilir.':
        print(f"ID: {aday['id_no']}, {aday['name']}, {aday['surname']}")
        
      
        veri_seti = pd.read_csv('Job_Placement_Data.csv', delimiter=';')
        
        # Sütun ekleyerek sahte bir 'İse_alindi_mi' oluşturun
        veri_seti['Ise_alindi_mi'] = np.random.choice([0, 1], size=len(veri_seti))

        # Label Encoding uygula
        label_encoder = LabelEncoder()
        veri_seti['work_experience'] = label_encoder.fit_transform(veri_seti['work_experience'])

        # Veriyi işleyin ve gerekli özellikleri seçin
        X = veri_seti[['science_exam_score', 'work_experience']]
        y = veri_seti['Ise_alindi_mi']

        # Veriyi eğitim ve test setlerine ayırın
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Lojistik Regresyon modelini oluşturun
        model = LogisticRegression()

        # Modeli eğitin
        model.fit(X_train, y_train)

        # Test seti üzerinde tahmin yapın
        y_pred = model.predict(X_test)

        # Doğruluk skorunu hesaplayın
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Modelin doğruluk skoru: {accuracy}')



