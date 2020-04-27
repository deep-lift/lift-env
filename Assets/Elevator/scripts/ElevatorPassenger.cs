
using System.Collections.Generic;
using UnityEngine;

public class ElevatorPassenger:  PoolObj<ElevatorPassenger>
{

    public int startFloor,destFloor;
    public float timeWaiting;

    public ElevatorPassenger()
    {
     
    }



    public void Init()
    {
        startFloor = destFloor = 0;
    }

    public void SetFloor(int start,int dest)
    {
        startFloor = start;
        destFloor = dest;

        timeWaiting = Time.fixedTime;

    }

}
