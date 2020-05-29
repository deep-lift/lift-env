using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class Buildfloor : MonoBehaviour
{


    public TextMeshPro[] textCallButton = new TextMeshPro[(int)MOVE_STATE.end];
    public TextMeshPro textPassenger;
    public TextMeshPro textFloor;

    Building building;


    public List<ElevatorPassenger> listPassenger = new List<ElevatorPassenger>();

    public List<ElevatorAgent> LandingElevators = new List<ElevatorAgent>();

    public ElevatorAgent[] callReservedEl = new ElevatorAgent[(int)MOVE_STATE.end];

    

    int floorNo;
    int passengerCount = 0;
   

    static float checkInterval = 1;

    float checkTime = 0;

	// Use this for initialization
	void Start ()
    {
        passengerCount = 0;

    }

    // Update is called once per frame
    void FixedUpdate ()
    {
        if ((Time.fixedTime-checkTime)>1f)
            ChkUpDownButton();

    }

    public void Init()
    {
        while( listPassenger.Count>0)
        {
            var p = listPassenger[0];
            p.Dispose();
            listPassenger.RemoveAt(0);

        }

        passengerCount = 0;
        textPassenger.text = passengerCount.ToString();

        SetButton(MOVE_STATE.Up, false);
        SetButton(MOVE_STATE.Down, false);

        checkTime = Time.fixedTime;
    }

    public int GetFloorNo()
    {
        return floorNo;
    }

    public int GetPassengerCount()
    {
        return passengerCount;
    }

    
    public bool IsCallRequest(MOVE_STATE dir)
    {
        if (textCallButton[(int)dir] == null
            || !textCallButton[(int)dir].gameObject.activeSelf)
            return false;


        return true;

    }

    public bool IsNoCall()
    {
        if (textCallButton[(int)MOVE_STATE.Down].gameObject.activeSelf
            || textCallButton[(int)MOVE_STATE.Up].gameObject.activeSelf)
            return false;

        return true;
    }

    public void SetFloor(int floor,Building building_)
    {
        building = building_;
        textFloor.text = floor.ToString();
        floorNo = floor;

    }



    public void SetPassenger(int passenger)
    {
        passengerCount = passenger;
        textPassenger.text = passengerCount.ToString();
    }

    public List<int> AddPassenger(int passenger)
    {
        passengerCount += passenger;
        textPassenger.text = passengerCount.ToString();

        List<int> destList = new List<int>();

        for (int i = 0; i < passenger; ++i)
        {
            ElevatorPassenger p = ElevatorPassenger.s_Pooler.Alloc();
            p.startFloor = floorNo;

            while(true)
            {
                p.destFloor = Random.Range(0,ElevatorAcademy.floors);
                if (p.destFloor != p.startFloor)
                {
                    destList.Add(p.destFloor);
                    break;
                }
            }           
            listPassenger.Add(p);
        }

        textPassenger.text = listPassenger.Count.ToString();

        ChkUpDownButton();

        return destList;
    }

    public void AddPassenger(List<int> destList)
    {
        passengerCount += destList.Count;
        textPassenger.text = passengerCount.ToString();

        for (int i = 0; i < destList.Count; ++i)
        {
            ElevatorPassenger p = ElevatorPassenger.s_Pooler.Alloc();
            p.startFloor = floorNo;
            p.destFloor = destList[i];
            listPassenger.Add(p);
        }

        textPassenger.text = listPassenger.Count.ToString();

        ChkUpDownButton();
    }


    public void SetButton(MOVE_STATE dir,bool bOn)
    {
        textCallButton[(int)dir].gameObject.SetActive(bOn);

        if (bOn)
        {
            building.CallRequest(floorNo, dir);

        }
    }

    public void ChkUpDownButton()
    {
        bool up = false, dn = false;

        foreach(var p in listPassenger)
        {
            if(p.destFloor> floorNo)
            {
                up = true;
            }
            else
            {
                dn = true;
            }
        }

        SetButton(MOVE_STATE.Up, up);
        SetButton(MOVE_STATE.Down, dn);

        checkTime = Time.fixedTime;

    }

    public void EnterElevator(ElevatorAgent el)
    {

        if (floorNo != (int)el.GetFloor() || !el.IsEnterableState())
            return;

        float delay=0;

        int idx = 0;
        while(idx< listPassenger.Count)
        {

            if (!el.IsEnterableState())
                break;

            if (el.AddPassenger(listPassenger[idx]))
            {
                listPassenger.RemoveAt(idx);
                delay += Random.Range(0.6f, 1.0f);
            }
            else
                ++idx;
            
        }


        textPassenger.text = listPassenger.Count.ToString();

       
        LandingElevators.Add(el);
        return;

    }

   

   

}



