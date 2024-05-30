# speech
from gtts import gTTS   #Được sử dụng để tạo giọng nói từ văn bản.
import playsound   #Thư viện phát âm thanh trong Python, thường được sử dụng để phát các tệp âm thanh.
import speech_recognition as sr  #Được sử dụng để nhận dạng giọng nói.
from googletrans import Translator   #Được sử dụng để dịch văn bản qua dịch vụ Google Translate.
from tkinter import messagebox as msg 
import tkinter.messagebox as msg



# tkinter #Được sử dụng để tạo giao diện đồ họa người dùng (GUI).
import tkinter as tk
from tkinter import ttk
from tkinter import *


#EDA
import numpy as np   #Được sử dụng để làm việc với mảng và ma trận, thực hiện tính toán số học và khoa học.
import pandas as pd  #Được sử dụng để làm việc với dữ liệu dạng bảng và thực hiện các phép xử lý, phân tích dữ liệu
from scipy import stats   #Được sử dụng cho tính toán khoa học và kỹ thuật, bao gồm xử lý tín hiệu, thống kê, tối ưu hóa, đại số tuyến tính, vv.
from sklearn import preprocessing   #Mô-đun của scikit-learn cung cấp các công cụ tiền xử lý dữ liệu, thường được sử dụng để chuẩn hóa dữ liệu và xử lý các giá trị thiếu.
from sklearn.feature_selection import SelectKBest, chi2    #Mô-đun của scikit-learn cho việc lựa chọn đặc trưng, thường được sử dụng để chọn ra các đặc trưng quan trọng nhất trong dữ liệu.
import matplotlib.pyplot as plt   #Được sử dụng để tạo đồ thị và biểu đồ.
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mpl_toolkits.mplot3d   #Mô-đun cho việc tạo và hiển thị đồ thị 3D, thường được sử dụng khi làm việc với dữ liệu có ba hoặc nhiều hơn các biến độc lập.


#Frames
from PIL import Image, ImageTk   #Được sử dụng để xử lý hình ảnh. 
import cv2   #Thư viện xử lý hình ảnh và video trong Python, thường được sử dụng để xử lý hình ảnh và video
import time
from PIL import Image, ImageOps, ImageTk, ImageFilter
from tkinter import colorchooser   #dùng chọn màu sử dụng hộp thoại chọn màu.
from tkinter import filedialog   #Cung cấp các hộp thoại để người dùng chọn tệp tin hoặc thư mục từ hệ thống tệp tin.








import os  #Thư viện hệ điều hành trong Python, thường được sử dụng để thao tác với các tệp tin và thư mục trên hệ thống.


# game
import pygame   #Được sử dụng cho game development.
import re   # Được sử dụng để tìm kiếm, xử lý và thay thế các chuỗi ký tự theo mẫu được định nghĩa trước
import sys, random  #cung cấp các công cụ để tạo và làm việc với số ngẫu nhiên. Được sử dụng để tạo số ngẫu nhiên, hoán đổi dữ liệu, lựa chọn ngẫu nhiên và thực hiện các tác vụ liên quan đến ngẫu nhiên
from pygame.locals import *   #Chứa các hằng số và biến cục bộ cho Pygame.
from matplotlib import cm   # chứa các màu sắc và bản đồ màu (colormaps) được sử dụng để tạo các biểu đồ màu sắc


################################# VOICE #######################################
def Voice_47NTTL():
    thuylinh47 = tk.Tk()
    thuylinh47.title("21133051_Nguyễn Thị Thùy Linh_Voice")
    thuylinh47.geometry("600x320")
    thuylinh47.resizable(FALSE, FALSE)
    thuylinh47.configure(bg="aquamarine")

    list_1 = []
    languages = ["vi", "ja", "en", "zh-CN", "ko", "fr", "de", "it", "ru", "es"]

    def NTTL47voice_input():         
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio_data = r.record(source, duration = 3)
        try:
            lang = 'vi'
            speech_text = r.recognize_google(audio_data, language=lang)
            text_box.insert("end", speech_text + "\n")
            text_box.update_idletasks()
        except sr.UnknownValueError:
            msg.showerror("Lỗi", "Tôi không hiểu bạn nói gì!")
            
    def NTTL47voice_output():
        text = text_box.get("1.0", "end")
        lang = combo_box.get()
        # translate the text to the selected language
        translator = Translator()
        translated_text = translator.translate(text, src='auto', dest=lang).text
        temp = gTTS(text=translated_text, lang=lang)
        temp.save('./test.mp3')
        pygame.mixer.init()
        pygame.mixer.music.load('./test.mp3')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(60)
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        os.remove('./test.mp3')

    buttonDoc = tk.Button(thuylinh47,text = "Đọc", width=6, height=2, font=("Arial", 12), command = NTTL47voice_input, bg = 'lightcoral')
    buttonDoc.place(x=10, y=5)

    buttonNoi = tk.Button(thuylinh47,text="Nói", width=6, height=2, font=("Arial", 12), command = NTTL47voice_output, bg = 'lightcoral')
    buttonNoi.place(x=370,y=5)

    text_box = tk.Text(thuylinh47, height=10, width=53, font=("Arial", 14), wrap="word")
    text_box.place(x=6, y=70)

    combo_box = ttk.Combobox(thuylinh47, values=languages)
    combo_box.place(x=450, y=30)


    thuylinh47.mainloop()

#################################  EDA  #######################################
col_list = []
df = pd.read_csv('D:/G447NTTLinh_DAHP.PyPro_Topic/G447NTTLinh_Topic.csv')
df_test = df.loc[0:100, :]
def EDA_47NTTL():
    global df 
    
    df = df.drop(columns=['Year_Birth', 'Dt_Customer'],axis=1)
    df = df.dropna(how='any')
    z = np.abs(stats.zscore(df._get_numeric_data()))
    df = df[(z < 3).all(axis=1)]

    
    list_chart = ['bar', 'line', 'barh']
    list_col_of_data = ['Complain', 'MntFishProducts', 'MntMeatProducts' , 'MntWines', 'MntGoldProds ', 'NumDealsPurchases', 'NumCatalogPurchases','NumStorePurchases'  ,'NumWebPurchases' ,'NumWebVisitsMonth', 'Id', 'Edducation', 'Marital_Status', 'Incom', 'Kidhome', 'Teenhome', 'Recency', 'MntFruits', 'MntSweet']

    print(df)
    
    
    def THUYLINH47_draw_chart():
        global df_test
        global col_list
        global kind
        #df_test = None  # Khởi tạo df_test là None
        _47NTTL = tk.Toplevel()
        _47NTTL.wm_attributes('-topmost', True)
        _47NTTL.title(kind + ' chart')
        _47NTTL.geometry("600x360")  
        # Increase the window size
        fig = plt.Figure(figsize=(10, 5))  
        # Increase the figure size
        if kind == 'scatter3d':
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df_test[col_list[0]], df_test[col_list[1]], df_test[col_list[2]])
            ax.set_xlabel(col_list[0])
            ax.set_ylabel(col_list[1])
            ax.set_zlabel(col_list[2])
        else:
            ax = fig.add_subplot(111)
            ax.set_title(kind + ' chart')
            df_test[col_list].plot(kind=kind, sharex=True, ax=ax)
            ax.xaxis.set_ticks(ax.get_xticks()[::2])

        canvas = FigureCanvasTkAgg(fig, master=_47NTTL)
        canvas.draw()
        canvas.get_tk_widget().pack()
        _47NTTL.mainloop()

    def get_kind():
        global kind
        kind = listchart_1.get()

    def add_col(col):
        global col_list
        col_list = [col]
        if col not in col_list:
            col_list.append(col)
        listchart_2.set("")  # clear the combobox's value
        for widget in NTTLinh_47.grid_slaves(row=4, column=3):
            widget.grid_remove()
        label_text = '\n'.join(col_list)
        label = tk.Label(NTTLinh_47,text=label_text)
        label.place(x=20, y=390)
    print(df)


    def NTTL47_Buoc_1(df):
        # Thiết lập treeview là widget con của Canvas
        treeview = ttk.Treeview(canvas, columns=list(df.columns), show="headings")
        treeview.pack(side="left", fill="both", expand=False)
        canvas.create_window((0, 0), window=treeview, anchor="nw")    
        # Thiết lập canvas để tự động thay đổi kích thước khi treeview thay đổi kích thước
        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        treeview.bind("<Configure>", on_configure)    
        # Thêm các cột vào treeview
        for column in list(df.columns):
            treeview.heading(column, text=column)    
        # Thêm dữ liệu vào treeview
        for row in df.itertuples(index=False):
            treeview.insert("", "end", values=row)    # Create a treeview widget   
        num_rows, num_cols = df.shape 
        dolon.config(text=f"Số dòng: {num_rows}, Số cột: {num_cols}")


    def NTTL47_Buoc_2(df):
        # data.destroy()
        # dolon.destroy()
        text_box.delete("1.0",END)
        # Thiết lập treeview là widget con của Canvas
        treeview = ttk.Treeview(canvas, columns=list(df.columns), show="headings")
        treeview.pack(side="left", fill="both", expand=False)
        canvas.create_window((0, 0), window=treeview, anchor="nw")
        # Thêm các cột vào treeview
        # Create a new dataframe with the count of non-null values for each column
        count_df = pd.DataFrame(df.count(), columns=['Count'])
    # Sort the dataframe by count in ascending order
        count_df = count_df.sort_values('Count')
    # Insert the values into the treeview
        for index, row in count_df.iterrows():
            treeview.insert("", "end", values=[index, row['Count']])
            treeview.heading("#1", text="Column")
            treeview.heading("#2", text="Count")
    # Thiết lập canvas để tự động thay đổi kích thước khi treeview thay đổi kích thước
        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            treeview.bind("<Configure>", on_configure)
    # Xóa các cột không cần thiết
        df = df.drop(columns=['Year_Birth', 'Dt_Customer'],axis=1)
    # Hiển thị số dòng và số cột của DataFrame
        num_rows, num_cols = df.shape
        dolon.config(text=f"Số dòng: {num_rows}, Số cột: {num_cols}")

        
    def NTTL47_Buoc_3(df):
        text_box.delete("1.0",END)
        # Bước 4: Xử lý DÒNG dữ liệu NULL  
        # Removing null values (Xóa tất cả các dòng có giá trị null trong tập FRAME dữ liệu.) 
        df = df.drop(columns=['Year_Birth', 'Dt_Customer'],axis=1)
        df = df.dropna(how='any') 
        text_box.insert(END,"Sau khi xử lí, các dòng và cột còn lại là\n")
        text_box.insert(END,f' {df.shape}')
       # kiểm tra lại số lượng cột & dòng của df sau khi XL NULL các dòng DL


    def NTTL47_Buoc_4(df):
        text_box.delete("1.0",END)
        df = df.drop(columns=['Year_Birth', 'Dt_Customer'],axis=1)
        df = df.dropna(how='any') 
        #kiểm tra tập dữ liệu có bất kỳ ngoại lệ nào không 
        z = np.abs(stats.zscore(df._get_numeric_data()).to_numpy()) 
        # Dò tìm và lấy các giá trị cá biệt trong tập dữ liệu gốc thông qua điểm z (z_score) 
        text_box.insert(END,'\n    MA TRAN Z-SCORE\n\n') 
        text_box.insert(END,f'{z}') 
        # in ra tập (ma trận) các giá trị z-score từ tập dữ liệu gốc 
        df= df[(z < 3).all(axis=1)]
        # kiểm tra và chỉ giữ lại trong df các giá trị số liệu tưng ứng với z-score < 3  # {loại các giá trị >= 3} vì các giá trị z-score >=3 tướng ứng với số liệu quá khác biệt so với các số liệu còn lại (“cá biệt” = “ngoại lệ” = isolated} 
        text_box.insert(END,"\n\n  Số dòng & cột dữ liệu sau khu xử lý các giá trị cá biệt \n")
        text_box.insert(END,f'\n {df.shape}') 
       # xác định số dòng & cột dữ liệu sau khu xử lý các giá trị cá biệt 
       
        
    def NTTL47_Buoc_5(df):
        text_box.delete("1.0",END)
        # Thiết lập treeview là widget con của Canvas
        treeview = ttk.Treeview(canvas, columns=list(df.columns), show="headings")
        treeview.pack(side="left", fill="both", expand=False)
        canvas.create_window((0, 0), window=treeview, anchor="nw")
        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        treeview.bind("<Configure>", on_configure)  
        df = df.drop(columns=['Year_Birth', 'Dt_Customer'],axis=1)
        df = df.dropna(how='any') 
        #kiểm tra tập dữ liệu có bất kỳ ngoại lệ nào không 
        z = np.abs(stats.zscore(df._get_numeric_data()).to_numpy()) 
        df= df[(z < 3).all(axis=1)]
        df['Education'].replace({'Graduation': 1, 'PhD': 2, '2n Cycle': 3, 'Master': 4, 'Basic': 5},inplace = True) 
        df['Marital_Status'].replace({'Divorced': 1, 'Single': 2, 'Married': 3, 'Together': 4, 'Widow': 5, 'YOLO': 6, 'Alone': 7, 'Absurd': 8},inplace = True) 
        for idx, column in enumerate(df.columns):
            treeview.heading(idx, text=column)
        for row in df.itertuples(index=False):
            treeview.insert("", "end", values=list(row))   # Create a treeview widget
        num_rows, num_cols = df.shape 
        dolon.config(text=f"Số dòng: {num_rows}, Số cột: {num_cols}")
        
        
    def NTTL47_Buoc_6(df):
        text_box.delete("1.0",END)
        df = df.drop(columns=['Year_Birth', 'Dt_Customer'],axis=1)
        df = df.dropna(how='any')    
        #kiểm tra tập dữ liệu có bất kỳ ngoại lệ nào không
        z = np.abs(stats.zscore(df._get_numeric_data()).to_numpy()) 
        df= df[(z < 3).all(axis=1)]
        df['Education'].replace({'Graduation': 1, 'PhD': 2, '2n Cycle': 3, 'Master': 4, 'Basic': 5},inplace = True) 
        df['Marital_Status'].replace({'Divorced': 1, 'Single': 2, 'Married': 3, 'Together': 4, 'Widow': 5, 'YOLO': 6, 'Alone': 7, 'Absurd': 8},inplace = True) 
        # -----Bước 7: Chuẩn hóa (Rời rạc hóa) tập dữ liệu Input dùng ..MaxMin 
        # CHUẨN HÓA DL 
        scaler = preprocessing.MinMaxScaler()
        numeric_columns = df.select_dtypes(include=np.number).columns
        scaler.fit(df[numeric_columns])
        df[numeric_columns] = scaler.transform(df[numeric_columns])
       # Thiết lập treeview là widget con của Canvas
        treeview = ttk.Treeview(canvas, columns=list(df.columns), show="headings")
        treeview.pack(side="left", fill="both", expand=False)
        canvas.create_window((0, 0), window=treeview, anchor="nw")
        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        treeview.bind("<Configure>", on_configure)
        # Thêm các cột vào treeview
        for column in list(df.columns):
            treeview.heading(column, text=column)
        # Thêm dữ liệu vào treeview
        for row in df.itertuples(index=False):
            treeview.insert("", "end", values=row)    # Create a treeview widget
        num_rows, num_cols = df.shape 
        dolon.config(text=f"Số dòng: {num_rows}, Số cột: {num_cols}")


    def NTTL47_Buoc_7(df):
    # Bước 8: Nạp các thuộc tính quan trọng vào Dataset 
    #The important features are put in a data frame 
        text_box.delete("1.0",END)
        df = df.drop(columns=['Year_Birth', 'Dt_Customer'],axis=1) 
        df = df.dropna(how='any') 
        z = np.abs(stats.zscore(df._get_numeric_data()).to_numpy()) 
        df= df[(z < 3).all(axis=1)]
        df['Education'].replace({'Graduation': 1, 'PhD': 2, '2n Cycle': 3, 'Master': 4, 'Basic': 5},inplace = True) 
        df['Marital_Status'].replace({'Divorced': 1, 'Single': 2, 'Married': 3, 'Together': 4, 'Widow': 5, 'YOLO': 6, 'Alone': 7, 'Absurd': 8},inplace = True) 
        scaler = preprocessing.MinMaxScaler()
        numeric_columns = df.select_dtypes(include=np.number).columns
        scaler.fit(df[numeric_columns])
        df[numeric_columns] = scaler.transform(df[numeric_columns])
        #Bước 8: Xác định mô hình trích lọc các thuộc tính đặc trưng: EDA  
        X = df.loc[:,df.columns!='Response']  # xác định tập DL Input (X) = All trừ (chú ý !) cột DL đoán đầu ra RainTomorrow 
        y = df[['Response']] # xác định tập DL ra 
        selector = SelectKBest(chi2, k=3)# sd các hàm ... trong thư viện sklearn = Mô hình xác định các Thuộc tính quan trọng quyết định việc dự đoàn DL output = trích lọc Đặc trưng = Feature Extraction 
        selector.fit(X, y) # Áp dụng mô hình trên vào .... 
        X_new = selector.transform(X) # Chuyên DL Input teho mô hình 
        text_box.insert(END, f" {X_new}") 
        text_box.insert(END, f"\n\n{ y}") 
        text_box.insert(END,f"\n\n{X.columns[selector.get_support(indices=True)]}") 


    def NTTL47_Buoc_8(df):    
        text_box.delete("1.0",END)
        df = df.drop(columns=['Year_Birth', 'Dt_Customer'],axis=1) 
        df = df.dropna(how='any') 
        z = np.abs(stats.zscore(df._get_numeric_data()).to_numpy()) 
        df= df[(z < 3).all(axis=1)]
        df['Education'].replace({'Graduation': 1, 'PhD': 2, '2n Cycle': 3, 'Master': 4, 'Basic': 5},inplace = True) 
        df['Marital_Status'].replace({'Divorced': 1, 'Single': 2, 'Married': 3, 'Together': 4, 'Widow': 5, 'YOLO': 6, 'Alone': 7, 'Absurd': 8},inplace = True) 
        scaler = preprocessing.MinMaxScaler()
        numeric_columns = df.select_dtypes(include=np.number).columns
        scaler.fit(df[numeric_columns])
        df[numeric_columns] = scaler.transform(df[numeric_columns])
        Y = df[['MntWines', 'MntMeatProducts', 'NumCatalogPurchases', 'Response']] 
       # Thiết lập treeview là widget con của Canvas
        treeview = ttk.Treeview(canvas, columns=list(df.columns), show="headings")
        treeview.pack(side="left", fill="both", expand=False)
        canvas.create_window((0, 0), window=treeview, anchor="nw")
        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        treeview.bind("<Configure>", on_configure)
        for idx, column in enumerate(Y.columns):
            treeview.heading(idx, text=column) 
        # Thêm dữ liệu vào treeview
        for row in Y.itertuples(index=False):
            treeview.insert("", "end", values=row)    # Create a treeview widget   
        num_rows, num_cols = Y.shape 
        dolon.config(text=f"Số dòng: {num_rows}, Số cột: {num_cols}")
        

    def NTTL47_Buoc_9(df):
        text_box.delete("1.0",END)
        df = df.drop(columns=['Year_Birth', 'Dt_Customer'],axis=1) 
        df = df.dropna(how='any') 
        z = np.abs(stats.zscore(df._get_numeric_data()).to_numpy()) 
        df= df[(z < 3).all(axis=1)]
        df['Education'].replace({'Graduation': 1, 'PhD': 2, '2n Cycle': 3, 'Master': 4, 'Basic': 5},inplace = True) 
        df['Marital_Status'].replace({'Divorced': 1, 'Single': 2, 'Married': 3, 'Together': 4, 'Widow': 5, 'YOLO': 6, 'Alone': 7, 'Absurd': 8},inplace = True) 
        scaler = preprocessing.MinMaxScaler()
        numeric_columns = df.select_dtypes(include=np.number).columns
        scaler.fit(df[numeric_columns])
        df[numeric_columns] = scaler.transform(df[numeric_columns])
        df = df[['MntWines', 'MntMeatProducts', 'NumCatalogPurchases', 'Response']] 
        treeview = ttk.Treeview(canvas, columns=list(df.columns), show="headings")
        treeview.pack(side="left", fill="both", expand=False)
        canvas.create_window((0, 0), window=treeview, anchor="nw")
        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        treeview.bind("<Configure>", on_configure)
       #Bước 10: EDA theo nhu cầu thực tế => input vào các mô hình AI, ML,... 
       # Đơn giản nhất là lấy 1 thuộc tính đầu vào (Humidity3pm) để XD Mô hình
        X = df[['MntMeatProducts','Response']] 
        for idx, column in enumerate(X.columns):
            treeview.heading(idx, text=column) 
        # Thêm dữ liệu vào treeview
        for row in X.itertuples(index=False):
            treeview.insert("", "end", values=row)    # Create a treeview widget   
        num_rows, num_cols = X.shape 
        dolon.config(text=f"Số dòng: {num_rows}, Số cột: {num_cols}")
        
           
    def voice_input():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio_data = r.record(source, duration=3)
        try:
            df = pd.read_csv('D:/G447NTTLinh_DAHP.PyPro_Topic/G447NTTLinh_Topic.csv')
            # Convert the speech to text
            text = r.recognize_google(audio_data, language='vi')
            # Update the label text
           # label.config(text=text)
            # Perform actions based on the recognized text
            if text == "Bước 1" or text == "bước 1":
                NTTL47_Buoc_1(df)
            elif text == "Bước 2" or text == "bước 2":
                NTTL47_Buoc_2(df)       
            elif text == "Bước 3" or text == "bước 3":
                NTTL47_Buoc_3(df)            
            elif text == "Bước 4" or text == "bước 4":
                NTTL47_Buoc_4(df)
            elif text == "Bước 5" or text == "bướcc 5":
                NTTL47_Buoc_5(df)
            elif text == "Bước 6" or text == "bước 6":
                NTTL47_Buoc_6(df)
            elif text == "Bước 7" or text == "bước 7":
                NTTL47_Buoc_7(df)
            elif text == "Bước 8" or text == "bước 8":
                NTTL47_Buoc_8(df)
            elif text == "Bước 9" or text == "bước 9":
                NTTL47_Buoc_9(df)
            else:
                # Handle unrecognized text
                print("Không nhận dạng được lệnh: " + text)
        except sr.UnknownValueError:
            msg.showerror("Lỗi", "Tôi không hiểu bạn nói gì!")

    col_list = []
    list_chart = ['bar', 'line', 'barh']
    list_col_of_data = ['Complain', 'MntFishProducts', 'MntMeatProducts' , 'MntWines', 'MntGoldProds ', 'NumDealsPurchases', 'NumCatalogPurchases','NumStorePurchases'  ,'NumWebPurchases' ,'NumWebVisitsMonth', 'Id', 'Edducation', 'Marital_Status', 'Incom', 'Kidhome', 'Teenhome', 'Recency', 'MntFruits', 'MntSweet']


    # Tạo windowform
    NTTLinh_47 = tk.Tk()
    NTTLinh_47.title("  21133051_Nguyễn Thị Thùy Linh_EDA  ")
    NTTLinh_47.geometry("1200x600")
    NTTLinh_47.configure(background='aquamarine')

    # Thêm một nhãn
    label_1 = tk.Label(NTTLinh_47, text="Mời bạn đọc chọn bước", font=("Time New Roman", 12))
    label_1.place(x= 20, y = 15)

    # Thêm một nút
    button_2 = tk.Button(NTTLinh_47, text=" Chọn bước ", bg="deeppink", command=voice_input)
    button_2.place(x= 35, y =68)

    # Thêm một text
    text_box = tk.Text(NTTLinh_47, font=("Time New Roman", 12), height = 34, width = 65)
    text_box.place(x=200, y=20)

    # Thêm một nhãn
    label_3 = tk.Label(NTTLinh_47, text="Biểu đồ ", font=("Time New Roman", 12))
    label_3.place(x= 20, y = 140)

    # Quy định vẽ từ 1-> 3 cột
    label_4 = tk.Label(NTTLinh_47, text='* Loại: ', font=("Time New Roman", 12))
    label_4.place(x=20, y =230)

    listchart_1 = ttk.Combobox(NTTLinh_47, values=list_chart)
    listchart_1.grid(padx=20, pady=270, sticky=tk.W)
    listchart_1.bind("<<ComboboxSelected>>", lambda event: get_kind())

    label_5 = tk.Label(NTTLinh_47, text='* Cột: ', font=("Time New Roman", 12))
    label_5.place(x=20, y =310)
    listchart_2 = ttk.Combobox(NTTLinh_47, values=list_col_of_data)
    listchart_2.place(x=20, y=350)
    listchart_2.bind("<<ComboboxSelected>>", lambda event: add_col(listchart_2.get()))

    # Tạo text để hiển thị data preview
    data = tk.Text(NTTLinh_47 ,borderwidth = 4, width = 55, height = 35)
    data.pack_propagate(False)
    data.place(x=800, y=20)

    # Tạo một thanh cuộn ngang cho canvas
    hscrollbar = tk.Scrollbar(data, orient="horizontal")
    hscrollbar.pack(side="top", fill="x")

    # Tạo một thanh cuộn dọc cho canvas
    vscrollbar = tk.Scrollbar(data, orient="vertical")
    vscrollbar.pack(side="left", fill="y")
        
     # Tạo Canvas để chứa treeview và thanh cuộn
    canvas = tk.Canvas(data, width=320, height=60, xscrollcommand=hscrollbar.set, yscrollcommand=vscrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    hscrollbar.config(command=canvas.xview)
    vscrollbar.config(command=canvas.yview)

    dolon = tk.Label(NTTLinh_47, text="", relief = tk.SUNKEN, borderwidth = 4, width = 30, height = 2)
    dolon.place(x=1030, y=600)

    button_3 = tk.Button(NTTLinh_47, text="  **Kết quả**  ", command=THUYLINH47_draw_chart, bg="deeppink")
    button_3.place(x=35, y =180)


    # Khởi chạy main event loop
    NTTLinh_47.mainloop()


#################################  GAME  ######################################
def Game_47NTTL():
    
    ####################################
    #PHẦN 1: ĐỊNH NGHĨA CÁC THAM SỐ ##
    #####################################
    ###KÍCH THƯỚC KHUNG MÀN HÌNH GAME
    WINDOWWIDTH = 400
    WINDOWHEIGHT = 600
    
    ###KHỞI TẠO THƯ VIỆN ĐỂ DÙNG
    pygame.init()
    
    ##TỐC ĐỘ KHUNG HÌNH CỦA VIDEO
    FPS = 60 # Famres Per Second
    fpsClock = pygame.time.Clock() #Lặp theo nhịp clock (tham số FPS) 
    
    ####################################
    #####PHẦN 2: NỀN GAME ##############
    #####################################
    
    #TỐC ĐỘ CUỘN NỀN
    BGSPEED = 2 # tốc độ cuộn nền
    BGIMG = pygame.image.load('D:/G447NTTLinh_DAHP.PyPro_Topic/G447NTTLinh_Topic/background.png') # hình nền
    
    # LAYER (SURFACE) NỀN
    DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
    pygame.display.set_caption(' 21133051_Nguyễn Thị Thùy Linh_Game ĐUA XE')
    
    # LỚP HÌNH NỀN = CUỘN NỀN
    class NTTL47_Background():
     def __init__(self):
         self.x = 0
         self.y = 0
         self.speed = BGSPEED
         self.img = BGIMG
         self.width = self.img.get_width()
         self.height = self.img.get_height()
    
     def draw(self):
         DISPLAYSURF.blit(self.img, (int(self.x), int(self.y)))
         DISPLAYSURF.blit(self.img, (int(self.x), int(self.y-self.height)))
     def update(self):
         self.y += self.speed
         if self.y > self.height:
             self.y -= self.height
    
    #####PHẦN 3: XE TRONG GAME #########
    """
    • X_MARGIN là lề hai bên trái và phải (xe không được vượt qua đó).
    • CARWIDTH và CARHEIGHT là kích thước của xe.
    • CARSPEED là tốc độ di chuyển (tiến, lùi, trái, phải) của xe.
    • CARIMG là ảnh chiếc xe.
    """
    #####################################
    
    #KÍCH THƯỚC XE
    X_MARGIN = 80
    CARWIDTH = 40
    CARHEIGHT = 60
    CARSPEED = 3
    CARIMG = pygame.image.load('D:/G447NTTLinh_DAHP.PyPro_Topic/G447NTTLinh_Topic/car.png')
    
    #LỚP XE TRONG GAME
    class Car():
     def __init__(self):
         self.width = CARWIDTH
         self.height = CARHEIGHT
         self.x = (WINDOWWIDTH-self.width)/2
         self.y = (WINDOWHEIGHT-self.height)/2
         self.speed = CARSPEED
         self.surface = pygame.Surface((self.width, self.height))
         self.surface.fill((255, 255, 255))
     def draw(self):
         DISPLAYSURF.blit(CARIMG, (int(self.x), int(self.y)))
     def update(self, moveLeft, moveRight, moveUp, moveDown):
         if moveLeft == True:
             self.x -= self.speed
         if moveRight == True:
             self.x += self.speed
         if moveUp == True:
             self.y -= self.speed
         if moveDown == True:
             self.y += self.speed
     
         if self.x < X_MARGIN:
             self.x = X_MARGIN
         if self.x + self.width > WINDOWWIDTH - X_MARGIN:
             self.x = WINDOWWIDTH - X_MARGIN - self.width
         if self.y < 0:
             self.y = 0
         if self.y + self.height > WINDOWHEIGHT :
             self.y = WINDOWHEIGHT - self.height
             
    ####################################
    #PHẦN 4: XE CHƯỚNG NGẠI VẬT = XE NGƯỢC CHIỀU:obstacles ##
    """
    • LANEWIDTH là độ rộng của 1 làn xe (đường có 4 làn).
    • DISTANCE là khoảng cách giữa các xe theo chiều dọc.
    • OBSTACLESSPEED là tốc độ ban đầu của những chiếc xe.
    • CHANGESPEED dùng để tăng tốc độ của những chiếc xe theo thời gian.
    • OBSTACLESIMG là ảnh chiếc xe.
    """
    #####################################
    
    LANEWIDTH = 60
    DISTANCE = 200
    OBSTACLESSPEED = 2
    CHANGESPEED = 0.001
    OBSTACLESIMG = pygame.image.load('D:/G447NTTLinh_DAHP.PyPro_Topic/G447NTTLinh_Topic/obstacles.png')   
    class Obstacles():
     def __init__(self):
         self.width = CARWIDTH
         self.height = CARHEIGHT
         self.distance = DISTANCE
         self.speed = OBSTACLESSPEED
         self.changeSpeed = CHANGESPEED
         self.ls = []
         for i in range(5):
             y = -CARHEIGHT-i*self.distance
             lane = random.randint(0, 3)
             self.ls.append([lane, y])
     def draw(self):
         for i in range(5):
             x = int(X_MARGIN + self.ls[i][0]*LANEWIDTH + (LANEWIDTH-self.width)/2)
             y = int(self.ls[i][1])
             DISPLAYSURF.blit(OBSTACLESIMG, (x, y))
     def update(self):
         for i in range(5):
             self.ls[i][1] += self.speed
         self.speed += self.changeSpeed
         if self.ls[0][1] > WINDOWHEIGHT:
             self.ls.pop(0)
             y = self.ls[3][1] - self.distance
             lane = random.randint(0, 3)
             self.ls.append([lane, y])
             
    ####################################
    #PHẦN 5: TÍNH ĐIỂM ##
    #####################################
    
    class NTTL47_Score():
        def __init__(self):
            self.score = 0
        def draw(self):
            font = pygame.font.SysFont('consolas', 30)
            scoreSuface = font.render('Score: '+str(int(self.score)), True, (0, 0, 0))
            DISPLAYSURF.blit(scoreSuface, (10, 10))
        def update(self):
            self.score += 0.02
            
    ####################################
    #PHẦN 6: XỬ LÝ VA CHẠM: Collision ##
    #####################################
    
    def rectCollision(rect1, rect2):
     if rect1[0] <= rect2[0]+rect2[2] and rect2[0] <= rect1[0]+rect1[2] and rect1[1] <= rect2[1]+rect2[3] and rect2[1] <= rect1[1]+rect1[3]:                
        return True
     return False
    def isGameover(car, obstacles):
     carRect = [car.x, car.y, car.width, car.height]
     for i in range(5):
         x = int(X_MARGIN + obstacles.ls[i][0]*LANEWIDTH + (LANEWIDTH-obstacles.width)/2)
         y = int(obstacles.ls[i][1])
         obstaclesRect = [x, y, obstacles.width, obstacles.height]
         if rectCollision(carRect, obstaclesRect) == True:
             return True
     return False
    
    ####################################
    #PHẦN 7: CÁC THỦ TỤC CHƠI GAME ##
    """
    • gameStart() là phần chuẩn bị khi vừa mở game lên.
    • gamePlay() là phần chơi chính.
    • gameOver() là phần xuất hiện khi thua 1 màn chơi.
    """
    #####################################
    
    def gameOver(bg, car, obstacles, score):
     font = pygame.font.SysFont('consolas', 60)
     headingSuface = font.render('GAMEOVER', True, (255, 0, 0))
     headingSize = headingSuface.get_size()
     font = pygame.font.SysFont('consolas', 20)
     commentSuface = font.render('Press "space" to replay', True, (0, 0, 0))
     commentSize = commentSuface.get_size()
     while True:
         for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 pygame.quit()
                 sys.exit()
             if event.type == pygame.KEYUP:
                 if event.key == K_SPACE:
                     return
         bg.draw()
         car.draw()
         obstacles.draw()
         score.draw()
         DISPLAYSURF.blit(headingSuface, (int((WINDOWWIDTH - headingSize[0])/2), 100))
         DISPLAYSURF.blit(commentSuface, (int((WINDOWWIDTH - commentSize[0])/2), 400))
         pygame.display.update()
         fpsClock.tick(FPS)
         
    def gameStart(bg):
     bg.__init__()
     font = pygame.font.SysFont('consolas', 60)
     headingSuface = font.render('RACING', True, (255, 0, 0))
     headingSize = headingSuface.get_size()
     font = pygame.font.SysFont('consolas', 20)
     commentSuface = font.render('Press "space" to play', True, (0, 0, 0))
     commentSize = commentSuface.get_size()
     while True:
         for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 pygame.quit()
                 sys.exit()
             if event.type == pygame.KEYUP:
                 if event.key == K_SPACE:
                     return
         bg.draw()
         DISPLAYSURF.blit(headingSuface, (int((WINDOWWIDTH - headingSize[0])/2), 100))
         DISPLAYSURF.blit(commentSuface, (int((WINDOWWIDTH - commentSize[0])/2), 400))
         pygame.display.update()
         fpsClock.tick(FPS)
         
    def gamePlay(bg, car, obstacles, score):
     car.__init__()
     obstacles.__init__()
     bg.__init__()
     score.__init__()
     moveLeft = False
     moveRight = False
     moveUp = False
     moveDown = False
     while True:
         for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 pygame.quit()
                 sys.exit()
             if event.type == KEYDOWN:
                 if event.key == K_LEFT:
                     moveLeft = True
                 if event.key == K_RIGHT:
                     moveRight = True
                 if event.key == K_UP:
                     moveUp = True
                 if event.key == K_DOWN:
                     moveDown = True
             if event.type == KEYUP:
                 if event.key == K_LEFT:
                     moveLeft = False
                 if event.key == K_RIGHT:
                     moveRight = False
                 if event.key == K_UP:
                     moveUp = False
                 if event.key == K_DOWN:
                     moveDown = False    
         if isGameover(car, obstacles):
            return
         bg.draw()
         bg.update()
         car.draw()
         car.update(moveLeft, moveRight, moveUp, moveDown)
         obstacles.draw()
         obstacles.update()
         score.draw()
         score.update()
         pygame.display.update()
         fpsClock.tick(FPS)
         
    ####################################
    #PHẦN 8: HÀM MAIN ##
    #####################################
    
    def main():
     bg = NTTL47_Background()
     car = Car()
     obstacles = Obstacles()
     score = NTTL47_Score()
     gameStart(bg)
     while True:
         gamePlay(bg, car, obstacles, score)
         gameOver(bg, car, obstacles, score)
    if __name__ == '__main__':
        main()
    
        



#################################  FRAMES  ####################################
def Frames_47NTTL():
    # Khởi tạo đối tượng Form
    NTTL47 = tk.Tk() 
    NTTL47.title("47-NguyenThiThuyLinh-21133051") 
    NTTL47.geometry("600x600") 
    NTTL47.configure(background='aquamarine')
    NTTL47.resizable(0, 0)

    # Khởi tạo các thành phần trên Form
    link_label = Label(NTTL47, text="Link Video")
    link_label.place(x = 75, y = 10)

    link_entry = Entry(NTTL47, width=30)
    link_entry.place(x = 70,y = 40)

    select_button = Button(NTTL47, text="Chọn",background='darkred',foreground= 'lightyellow', command=lambda: OpenTextFile())
    select_button.place(x = 17, y = 37)

    file_label = Label(NTTL47, text="Lưu files")
    file_label.place(x = 400, y = 10)

    file_entry = Entry(NTTL47, width=30)
    file_entry.place(x = 400,y = 40)

    file_select_button = Button(NTTL47, text="Chọn",background='darkred',foreground= 'lightyellow', command=lambda: OpenFolder())
    file_select_button.place(x = 346, y = 37)

    cut_button = Button(NTTL47, text="   Cắt   ",background='darkred',foreground= 'lightyellow', command=lambda: CutVideo())
    cut_button.place(x=50 , y=120)

    xuly_button = Button(NTTL47, text="Chỉnh sửa ảnh",background='darkred',foreground= 'lightyellow', command=lambda: Image_fixing())
    xuly_button.place(x=50 , y=520)

    output_label = Label(NTTL47, text="Cropped frames:")
    output_label.place(x = 245, y = 125)

    output_text = Text(NTTL47, height=22, width=60)
    output_text.place(x=50, y=150)

    # Hàm mở tệp văn bản
    def OpenTextFile():     
        global filepath 
        filepath = filedialog.askopenfilename(title = "Select a video file", filetypes = (("MP4 files", "*.mp4"), ("AVI files", "*.avi")))
        link_entry.delete(0, END)
        link_entry.insert(0, filepath)

    def OpenFolder():
        global folderpath
        folderpath = filedialog.askdirectory()
        file_entry.delete(0, END)
        file_entry.insert(0, folderpath)

    def CutVideo():
        cap = cv2.VideoCapture(link_entry.get())
        count = 0
        output_folder = file_entry.get()
        
        # create new top-level window for entering start and end times
        cut_window = Toplevel(NTTL47)
        cut_window.geometry("300x150")
        cut_window.title("Cắt Video")
        cut_window.configure(background='palevioletred')
        
        # create label and entry for start time
        start_label = Label(cut_window, text="Bắt đầu:", bg = 'tomato')
        start_label.place(x=10, y=50)
        
        start_entry = Entry(cut_window)
        start_entry.place(x=140, y=50)
        
        # create label and entry for end time
        end_label = Label(cut_window, text="Kết thúc: ", bg = 'tomato')
        end_label.place(x =10, y =90)
        end_entry = Entry(cut_window)
        end_entry.place(x=140, y= 90)
        
        # create button to confirm time selection and start video cutting
        confirm_button = Button(cut_window, text="Xử lý", bg = 'tomato',command=lambda: CutVideoHelper(cap, count, output_folder, start_entry.get(), end_entry.get()))
        confirm_button.place(x=10, y= 10)
        
        cut_window.mainloop()
        
    def CutVideoHelper(cap, count, output_folder, start_time, end_time):
        start_time = float(start_time)
        end_time = float(end_time)

        # set the starting frame
        start_frame = int(start_time * cap.get(cv2.CAP_PROP_FPS))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # calculate the current time in seconds
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            # check if the current time is past the end time
            if current_time > end_time:
                break
            output_path = os.path.join(output_folder, "frame%d.jpg" % count)  # include the output_folder in the output_path
            cv2.imwrite(output_path, frame)
            output_text.insert(END, output_path + "\n")
            count += 1

        cap.release()
        output_text.insert(END, "Video cropped from %d to %d seconds.\n" % (start_time, end_time))

    def Image_fixing():
        win = tk.Toplevel()
        win.geometry("1150x600")
        win.title("XỬ LÝ ẢNH")
        win.config(bg="skyblue")

        pen_color = "black"
        pen_size = 10
        file_path = ""

        def add_image():
            nonlocal file_path
            file_path = filedialog.askopenfilename(
                title="Select Image File", filetypes=[("PNG Files", "*.png"), ("JPG Files", "*.jpg"), ("All Files", "*.*")], initialdir=os.getcwd())
            image = Image.open(file_path)
            width, height = int(image.width / 2), int(image.height / 2)
            image = image.resize((width, height), Image.LANCZOS)
            canvas.config(width=image.width, height=image.height)
            image = ImageTk.PhotoImage(image)
            canvas.image = image
            canvas.create_image(0, 0, image=image, anchor="nw")

        def change_color():
            nonlocal pen_color
            pen_color = colorchooser.askcolor(title="Select Pen Color")[1]

        def change_size(size):
            nonlocal pen_size
            pen_size = size

        def draw(event):
            x1, y1 = (event.x - pen_size), (event.y - pen_size)
            x2, y2 = (event.x + pen_size), (event.y + pen_size)
            canvas.create_oval(x1, y1, x2, y2, fill=pen_color, outline='')

        def clear_canvas():
            canvas.delete("all")
            canvas.create_image(0, 0, image=canvas.image, anchor="nw")

        def apply_filter(filter):
            nonlocal file_path
            image = Image.open(file_path)
            width, height = int(image.width / 2), int(image.height / 2)
            image = image.resize((width, height), Image.LANCZOS)
            if filter == "Black and White":
                image = ImageOps.grayscale(image)
            elif filter == "Blur":
                image = image.filter(ImageFilter.BLUR)
            elif filter == "Sharpen":
                image = image.filter(ImageFilter.SHARPEN)
            elif filter == "Smooth":
                image = image.filter(ImageFilter.SMOOTH)
            elif filter == "Emboss":
                image = image.filter(ImageFilter.EMBOSS)
            image = ImageTk.PhotoImage(image)
            canvas.image = image
            canvas.create_image(0, 0, image=image, anchor="nw")
            
        def save_image():
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", ".png"), ("All Files", ".*")])
            if file_path:
                canvas.postscript(file=file_path + '.eps')
                img = Image.open(file_path + '.eps')
                img.save(file_path, "png")
                img.close()
                os.remove(file_path + '.eps')
                msg.showinfo("Success", "   Đã lưu   ")

        save_button = tk.Button(win, text="  ***Lưu***  ",bg="salmon", command=save_image)
        save_button.pack(side="left", padx=5, pady=5)  

        left_frame = tk.Frame(win, width=200, height=600, bg="skyblue")
        left_frame.pack(side="left", fill="y")

        canvas = tk.Canvas(win, width=750, height=600)
        canvas.pack()

        image_button = tk.Button(left_frame, text=" Chọn ảnh ", command=add_image, bg="salmon")
        image_button.place(x=25 , y =15)

        color_button = tk.Button(left_frame, text="Thay đổi màu bút", command=change_color, bg="salmon")
        color_button.place(x=25 , y =155)

        pen_size_frame = tk.Frame(left_frame, bg="white")
        pen_size_frame.place(x=25 , y =205)

        pen_size_1 = tk.Radiobutton(pen_size_frame, text="Nhỏ", value=3, command=lambda: change_size(3), bg="skyblue")
        pen_size_1.pack(side="left")
        

        pen_size_2 = tk.Radiobutton(pen_size_frame, text="Vừa", value=5, command=lambda: change_size(5),bg="skyblue")
        pen_size_2.pack(side="left")
        pen_size_2.select()
        

        pen_size_3 = tk.Radiobutton(pen_size_frame, text="Lớn", value=7, command=lambda: change_size(7), bg="skyblue")
        pen_size_3.pack(side="left")
        

        clear_button = tk.Button(left_frame, text="Xóa màu vẽ", command=clear_canvas, bg="salmon")
        clear_button.place(x=25 , y =255)

        filter_label = tk.Label(left_frame, text="Chọn Filter", bg="skyblue")
        filter_label.place(x =25, y =70)
        filter_combobox = ttk.Combobox(left_frame, values=["Black and White", "Blur", "Emboss", "Sharpen", "Smooth"])
        filter_combobox.pack()
        filter_combobox.place(x =25, y =100)


        filter_combobox.bind("<<ComboboxSelected>>", lambda event: apply_filter(filter_combobox.get()))


        canvas.bind("<B1-Motion>", draw)

        win.mainloop()

    # Thực thi chương trình
    NTTL47.mainloop()

############################  THOÁT  ##########################################
def Thoat_47NTTL():
   traloi = msg.askquestion("Xác nhận","Thiệt thoát không (Y/N)?")
   if traloi == "yes": ThuyLinh.destroy()   
    

ThuyLinh = tk.Tk()
#Thêm tiêu đề cho cửa sổ
ThuyLinh.title("47_NTTLinh, LỚP21133C_HCMUTE, ĐỒ ÁN HỌC PHẦN: LẬP TRÌNH PYTHON, T5.2023")
#Đặt kích thước của cửa sổ
ThuyLinh.geometry('900x600')
#Không cho thay đổi size 
ThuyLinh.resizable(tk.FALSE, tk.FALSE)
ThuyLinh.configure(bg="lightblue")

t = "47_Nguyễn Thị Thùy Linh, LỚP 21133C_HCMUTE \n  ĐỒ ÁN HỌC PHẦN: LẬP TRÌNH PYTHON: EDA"
lblDT = tk.Label(ThuyLinh, text=t, background = "lightpink", fg = "blue", relief = tk.SUNKEN, font=("Arial Bold", 13), borderwidth = 3, width = 75, height = 3)
lblDT.place(x = 75, y = 15)

#Thêm một nút 
button_Voice = Button(ThuyLinh, text="VOICE", bg="blue", fg="white", font=("Times New Roman", 20), command= Voice_47NTTL)

#Thiết lập vị trí của nút nhấn có màu nền và màu chữ
button_Voice.place(x= 120, y=120)

#Thêm một nút nhấn Click Me
button_EDA = Button(ThuyLinh, text="  EDA  ", bg="blue", fg="white", font=("Times New Roman", 20), command = EDA_47NTTL)

#Thiết lập vị trí của nút nhấn có màu nền và màu chữ
button_EDA.place(x=310, y=120)

button_FRAMES = Button(ThuyLinh, text=" FRAMES ", bg="blue", fg="white", font=("Times New Roman", 20), command= Frames_47NTTL)
button_FRAMES.place(x=120, y=320)

button_GAME = Button(ThuyLinh, text=" GAME ", bg="blue", fg="white", font=("Times New Roman", 20), command = Game_47NTTL)
button_GAME.place(x=310, y=320)

#Thêm một nút 
button_Thoat = Button(ThuyLinh, text="THOÁT", bg="tomato", fg="white", font=("Times New Roman", 14), command = Thoat_47NTTL)
#Thiết lập vị trí của nút nhấn có màu nền và màu chữ
button_Thoat.place(x=120, y=500)


#Lặp vô tận để hiển thị cửa sổ
ThuyLinh.mainloop()
