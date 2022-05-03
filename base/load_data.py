# -*- coding: utf-8 -*-

import pymysql
import pandas as pd
from base import char_to_number


def load_from_tidb(host, port, user, passwd, db, sqlOrder, charTrans=False, charset="utf8"):
    '''
    从tidb数据库中读取数据
    
    Parameters
    ----------
    host      : IP地址，例如：117.50.32.96
    port      : 端口，例如：3306
    user      : 数据库用户名
    passwd    : 数据库密码
    db        : 数据库名称
    charTrans : 是否进行字符转换
    charset   : 字符类型
    
    Returns
    -------
    数据框
    '''
    try:
        db = pymysql.connect(host = host,
                             port = port,
                             user = user,
                             passwd = passwd,
                             db = db,
                             charset = charset)
        cur = db.cursor()
        print("数据库连接成功！")
    except:
        print("Can't connect to Tidb!")
        
    try:
        cur.execute(sqlOrder)
        data = cur.fetchall()
        index = cur.description
        names = [index[i][0].lower() for i in range(len(index))]
        df = pd.DataFrame(list(data),columns=names)
        print("数据已成功获取！")
        # 字符转换
        if charTrans == True:
            for var_item in df.dtypes[df.dtypes=='object'].index.tolist():
                df[var_item] = char_to_number(df[var_item])
            print("数据字符类型转换成功！")
    except:
        df = pd.DataFrame()
        print("No table loaded!")
    return df


def load_from_csv(filePath, fileSep=',', charTrans=False):
    '''
    从csv中读取数据
    filePath = r'F:\Python\Test\sample.csv'
    fileSep = ','
    tmp = load_from_csv(filePath).head()
    '''
    try:
        df = pd.read_csv(filePath,sep=fileSep,low_memory=False)
        # 字符转换
        if charTrans == True:
            for var_item in df.dtypes[df.dtypes=='object'].index.tolist():
                df[var_item] = char_to_number(df[var_item])
        
        return df
    except:
        print("Failed to read csv data!")


def load_from_sas(filePath, charTrans=False):
    '''
    从sas中读取数据
    '''
    try:
        df = pd.read_sas(filePath)
        # 字符转换
        if charTrans == True:
            for var_item in df.dtypes[df.dtypes=='object'].index.tolist():
                df[var_item] = char_to_number(df[var_item])
        return df
    except:
        print("Failed to read sas data!")




