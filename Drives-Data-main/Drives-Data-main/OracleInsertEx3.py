#Program inserting employee record in employee table
#OracleInsertEx2.py
import cx_Oracle
def recordinsert():
    while(True):
        try:
            con = cx_Oracle.connect("system/manager@localhost/xe")
            cur = con.cursor()
            #accept employee values from KBD
            print("-----------------------------------")
            empno=int(input("Enter Employee Number:"))
            ename=input("Enter Employee Name:")
            empsal=float(input("Enter Employee Salary:"))
            cname=input("Enter employee Comp Name:")
            print("-----------------------------------")
            #design the query
            iq="insert into employee values(%d,'%s',%f,'%s')"
            cur.execute(iq %(empno,ename,empsal,cname))
            con.commit()
            print("{} Employee Record Inserted--verify".format(cur.rowcount))
            print("--------------------------------------")
            ch=input("Do u want to insert another record(yes/no):")
            if(ch.lower()=="no"):
                print("thx for using this program")
                break
        except cx_Oracle.DatabaseError as kvr:
            print("Problem in Oracle DB:", kvr)
        except ValueError:
            print("Don't enter alnums,strs and symbols for empno and sal")

#main program
recordinsert()