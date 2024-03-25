#Program for Creating Employee Table
#OracleTableCreateEx1.py
import cx_Oracle # Step-1
def tablecreate():
    try:
        con=cx_Oracle.connect("system/manager@localhost/xe") # Step-2
        cur=con.cursor() # Step-3
        #Step-4
        tq="create table employee(eno number(2) primary key, name varchar2(10) not null,sal number(5,2) not null)"
        cur.execute(tq)
        #Step-5
        print("Employee Table created Sucessfully--verify")
    except cx_Oracle.DatabaseError as db:
        print("Problem in Oracle:",db)

#main program
tablecreate()