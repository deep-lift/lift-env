using System;
using System.Collections.Generic;
using System.Linq;

using System.IO;
using System.Threading.Tasks;






public class ClassPool<T> where T : new()
{
    Queue<T> qAlloc = new Queue<T>();
    Stack<T> rest = new Stack<T>();


    public T Alloc()
    {

        if (rest.Count > 0)
            return rest.Pop();

        T t = new T();

        qAlloc.Append(t);
        return t;
    }

    public void Reserve(int reserve)
    {

        for (int i = 0; i < reserve; ++i)
        {
            T t = new T();
            qAlloc.Append(t);

            rest.Append(t);
        }
    }

    public void Release(T t)
    {
        rest.Append(t);
    }


}

public class PoolObj<T> where T : new()
{
    public static ClassPool<T> s_Pooler;

    public static void InitPooler()
    {
        if (s_Pooler == null)
            s_Pooler = new ClassPool<T>();
    }


    ~PoolObj()
    {
        objectRemove(true);
    }

    private void objectRemove(bool self_call)
    {
        GC.SuppressFinalize(this);
        s_Pooler.Release((T)Convert.ChangeType(this, typeof(T)));
    }


    public void Dispose()
    {
        objectRemove(true);
    }

}




