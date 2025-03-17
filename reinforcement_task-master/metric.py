import pandas as pd
def decision_prices(test):
    
    test = test.set_index('dt') 
    tender_price = test['Цена на арматуру']
    decision = test['Объем']
    start_date = test.index.min()
    end_date = test.index.max()
    
    _results = []
    _active_weeks = 0
    for report_date in pd.date_range(start_date, end_date, freq='W-MON'):
        if _active_weeks == 0:  # Пришла пора нового тендера
            _fixed_price = tender_price.loc[report_date]
            _active_weeks = int(decision.loc[report_date])
        _results.append(_fixed_price)
        _active_weeks += -1
    cost = sum(_results)
    return cost # Возвращаем затраты на периоде