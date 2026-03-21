import datetime


print(datetime.datetime.now().date())
now_time = datetime.datetime.now().time()
now_time = str(now_time)
now_time_list = now_time.split(":")
print(now_time_list)
print(now_time_list[0]+"-"+now_time_list[1])