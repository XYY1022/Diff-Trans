def amplify(df_raw, length):
    for i in range(length):
        '''
        if 10 < df_raw.loc[i, 'PQJ'] < 20:
            df_raw.loc[i, 'PQJ'] *= 1.3
        if 10 < df_raw.loc[i, 'PWMP'] < 20:
            df_raw.loc[i, 'PWMP'] *= 1.3
        if 2500 < df_raw.loc[i, 'QWMP'] < 6000:
            df_raw.loc[i, 'QWMP'] *= 1.2
        '''
        if 1000 < df_raw.loc[i, 'QXX'] < 2000:
            df_raw.loc[i, 'QXX'] *= 1.3
        elif 2000 < df_raw.loc[i, 'QXX'] < 5000:
            df_raw.loc[i, 'QXX'] *= 1.4
        elif df_raw.loc[i, 'QXX'] > 5000:
            df_raw.loc[i, 'QXX'] *= 1.1

    return df_raw
