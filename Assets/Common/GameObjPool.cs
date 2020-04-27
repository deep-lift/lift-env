
using System.Collections.Generic;
using System.Collections;
using System.IO;
using UnityEngine;


public class GameObjPool: Dictionary<string, Queue<GameObject> >
{
    // static MonoBehaviour MonoObj = null;


    public Dictionary<string, Object> m_DicRes = new Dictionary<string, Object>();
    //public Dictionary<string, Queue<GameObject> > m_dicActiveObj = new Dictionary<string, Queue<GameObject>>();
    public string strDefaultDir;

     ~GameObjPool()
    {
        foreach(var item in this.Values)
        {
            while (item.Count > 0)
            {
               
                var obj= item.Dequeue();

                if (obj)
                    GameObject.Destroy(obj);
            }

        }

        this.Clear();

        m_DicRes.Clear();
    }


    public string LoadResource(string fileName,int nReserve =0)
    {
        string objName = Path.GetFileNameWithoutExtension(fileName);

       Object resObj = null;

        if (!m_DicRes.TryGetValue(objName, out resObj))
        {
            resObj = Resources.Load(fileName);

            if (resObj)
                m_DicRes.Add(objName, resObj);

            Queue<GameObject> list = null;

            GameObject obj = null ;

            if (!this.TryGetValue(objName, out list))
            {
                list = new Queue<GameObject>();
                Add(objName, list);
            }

            for (int i = 0; i < nReserve; ++i)
            {
                obj = (GameObject)GameObject.Instantiate(resObj);
                obj.name = objName;
                obj.SetActive(false);
                list.Enqueue(obj);
            }
        }
        else if(!resObj)
        {
            resObj = Resources.Load(fileName);
            m_DicRes.Remove(objName);

            if (resObj)
                m_DicRes.Add(objName, resObj);
        }

        if (!resObj)
            return null;

 
        return objName;
    }

    public string LoadResource(Object resObj, int nReserve = 0)
    {

        if (resObj == null)
            return null;

        string objName = Path.GetFileNameWithoutExtension(resObj.name);

        if (!m_DicRes.ContainsKey(objName))
        {
            m_DicRes.Add(objName, resObj);

            Queue<GameObject> list = null;

            GameObject obj = null;

            if (!this.TryGetValue(objName, out list))
            {
                list = new Queue<GameObject>();
                Add(objName, list);
            }

            for (int i = 0; i < nReserve; ++i)
            {
                obj = (GameObject)GameObject.Instantiate(resObj);
                obj.name = objName;
                obj.SetActive(false);
                list.Enqueue(obj);
            }
        }

        return objName;
    }

    public GameObject Alloc(string strName, MonoBehaviour monoObj,float lifeTime = 0f)
    {
        Queue<GameObject> list = null;

        strName = Path.GetFileNameWithoutExtension(strName);

        GameObject obj;

        if (this.TryGetValue(strName, out list))
        {
            while (list.Count > 0)
            {
                obj = list.Dequeue();

                if (obj)
                {
                    obj.SetActive(true);
                    if (lifeTime > 0f)
                    {
                        if (monoObj)
                            monoObj.StartCoroutine(ReleaseObj(obj, lifeTime));
                        else
                            GameObject.Destroy(obj, lifeTime);
                    }
                    return obj;
                }
            }
        }

        if(list==null)
             Add(strName, new Queue<GameObject>());

        Object resObj = null;
        if (!m_DicRes.TryGetValue(strName, out resObj)||resObj== null)
            return null;

        obj = (GameObject)GameObject.Instantiate(resObj);
        obj.name = strName;

        if (lifeTime > 0f)
        {
            if (monoObj)
                monoObj.StartCoroutine(ReleaseObj(obj, lifeTime));
            else
                GameObject.Destroy(obj, lifeTime);
        }
        return obj;
    }

    public void Release(GameObject obj)
    {
        if(!obj)
           return;

        Queue<GameObject> list = null;

        if (!this.TryGetValue(obj.name, out list))
        {
            GameObject.Destroy(obj);
            return;
        }

        obj.SetActive(false);
        list.Enqueue(obj);

       // Debug.Log("Name:" + obj.name + " Count:" + list.Count);
    }

    IEnumerator ReleaseObj(GameObject Obj,float time)
    {
        yield return new WaitForSeconds(time);

        Release(Obj);

        yield break;
    }

}

