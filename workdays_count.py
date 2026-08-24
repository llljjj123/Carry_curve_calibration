# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 13:44:42 2023

@author: LuJingjian
"""

class AccountDays:

    def workdays(start,end):
        '''
        计算两个日期间的工作日
        start:开始时间
        end:结束时间
        '''
        from datetime import datetime,timedelta
        from chinese_calendar import is_workday
         # 字符串格式日期的处理
        if type(start) == str:
            start = datetime.strptime(start,'%Y-%m-%d').date()
        if type(end) == str:
            end = datetime.strptime(end,'%Y-%m-%d').date()
        # 开始日期大，颠倒开始日期和结束日期
        if start > end:
            start,end = end,start
        counts = 0
        while True:
            if start > end:
                break
            if is_workday(start):
                counts += 1
            start += timedelta(days=1)
        return counts

    def tradedays(start,end):
        '''
        计算两个日期间的工作日
        start:开始时间
        end:结束时间
        '''
        from datetime import datetime,timedelta
        from chinese_calendar import is_holiday
        
            
        # 字符串格式日期的处理
        if type(start) == str:
            start = datetime.strptime(start,'%Y-%m-%d').date()
        if type(end) == str:
            end = datetime.strptime(end,'%Y-%m-%d').date()

        
        # 开始日期大，颠倒开始日期和结束日期
        if start > end:
            start,end = end,start
            
        counts = 0
        # 2024/2/9 is a work day but not a trade day
        # up to 2024/11/21 the package hasn't updated and remove 2024/2/9 from trade day yet
        if ((end >= date(2024,2,9))  & (start <= date(2024,2,9)) ):
            counts -= 1
        while True:
            if start > end:
                break
            if is_holiday(start) or start.weekday()==5 or start.weekday()==6:
                start += timedelta(days=1)
                continue
            counts += 1
            start += timedelta(days=1)
        
        return counts




    

if __name__ == '__main__':
    from datetime import date
    start_date = date(2025,8,4)
    end_date = date(2025,9,30)
    
    tradedays = AccountDays.tradedays
    workdays = AccountDays.workdays
    
    # if(end_date >= date(2024,2,9)):
    #     tradedays -= 1
    
    print(f'交易日数量：{tradedays(start_date,end_date)}') 
    print(f'工作日数量：{workdays(start_date,end_date)}') 
    print(f'{start_date.year}年交易日数量：{tradedays(date(2025,1,1), date(2025,12,31))}')