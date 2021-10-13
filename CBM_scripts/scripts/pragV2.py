import pandas as pd
#import data to a dictionary
sheets_dict = pd.read_excel('PT120.xlsx', sheet_name=None,usecols= 'A,B,E,F,I,J',skiprows=3)

#create a data frame
#full_table = pd.DataFrame()

#create dictionary d
d = {}
#iterate through sheets_dict
#drop NaN values
for name, sheet in sheets_dict.items():
    #print(name)
    #print(type(sheet))
    sheet = sheet.dropna()
    sheets_dict[name] = name
    d[name] = sheet
   


# for key, item in sheets_dict.items():
#     item = item.dropna()

#     sheets_dict[key] = item

# print(d.keys())
# print(d['120us_0.8s']['Time'])



#extract dictionary 1 by 1
for l in d.keys():
    #print(d[l])
    print(d[l][['Time.1','Intensity.1']])



#print(type(sheets_dict))
#print(type(d))
#full_table.to_csv('test.csv')

# import pandas as pd
# sheets_dict = pd.read_excel('Book1.xlsx', sheet_name=None)
# full_table = pd.DataFrame()
# for name, sheet in sheets_dict.items():
#     sheet['sheet'] = name
#     sheet = sheet.rename(columns=lambda x: x.split('\n')[-1])
#     full_table = full_table.append(sheet)
# full_table.reset_index(inplace=True, drop=True)
# print(full_table)