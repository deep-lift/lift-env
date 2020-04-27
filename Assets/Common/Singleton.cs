using UnityEngine;

public class Singleton<T> : MonoBehaviour where T : MonoBehaviour
{
    protected static T _instance = null;
    public static T Inst
    {
        get
        {
            if (_instance == null)
            {
                _instance = FindObjectOfType(typeof(T)) as T;

                if (_instance == null)
                {
                    Debug.Log("Nothing Singleton: " + typeof(T).FullName);
                    _instance = new GameObject(typeof(T).FullName).AddComponent<T>();
                }
            }
            return _instance;
        }
    }
}
