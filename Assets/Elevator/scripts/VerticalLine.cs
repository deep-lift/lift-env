using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VerticalLine : MonoBehaviour
{

 

    public GameObject callFloorBtn;
    public GameObject destFloorBtn;

    public LineRenderer line;

	// Use this for initialization
	void Start ()
    {
        SetHeight(ElevatorAcademy.height);
    }
	
	// Update is called once per frame
	void Update ()
    {
		
	}

    public void SetHeight(float height)
    {
        float z = line.GetPosition(0).z;

        line.SetPosition(0, new Vector3(0, height / 2f, z));
        line.SetPosition(1, new Vector3(0, -height / 2f, z));
    }

    public void SetCallResquest(bool bCall)
    {
        callFloorBtn.SetActive(bCall);
    }

    public void SetDestResquest(bool bCall)
    {
        destFloorBtn.SetActive(bCall);
    }
}
