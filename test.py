import os
import tushare as ts
pro = ts.pro_api('a9aaa1e9993b80434618160a797668ead539336bca943fc97d2b0d24')
pro._DataApi__http_url = "https://t.xiaodefa.top/"
df = pro.index_basic(limit=5)
# pro_bar接口请加上api=pro
df = ts.pro_bar(api=pro, ts_code="000001.SZ", limit=3)
print(df)