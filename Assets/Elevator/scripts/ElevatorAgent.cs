using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using MLAgents;

public enum MOVE_STATE:int
{
    Stop = 0,
    Down ,
    Up ,
    end,
}

public class ElevatorAgent : Agent
{
    public enum State
    {
        Ready,          //문닫고 멈춰있는 상태
        NormalMove,     //위아래 어느쪽이든 정상적인 이동상태
        Decelerate,     //다음층에 멈추기 위한 감속상태
        MoveStop,       //이동하는 중에 각층에 멈춤상태
        DoorOpening,    //문열는중
        DoorOpened,     //문열린 상태에서 승객내리고 타고
        DoorClosing,    //문닫히는 동안.
        Accelate,       //이동에 대한 가속상태
        Turn,
        End,
    };

    public enum Event
    {
        None,
        Call,               //각층에서 호출이 왔을 경우.
        DecelerateStart,     //이동중에 감속지점을 통과 했을때
        Arrived,            //각 층에 도착했을때
        DoorOpenRequest,    //문열기 요청
        DoorOpenEnd,        //문열기 끝
        DoorCloseStart,     //문닫기 시작
        DoorCloseEnd,       //문닫기 끝
        AccelateEnd,        //가속끝 정상속도 도달
        EmptyPassenger,     //승객이없다. 전부 내렸다.
        End

    }



    static GameObject resFloor;

    static Building building;

    static int[] moveDir = { 0, -1, 1 }; 

    public bool[] floorBtnflag;

    MOVE_STATE moveDirState;            //이동 상태방향 여부(-1,0,1) 아래,멈춤,위
    float currentFloor;                 //현재엘베가 있는 층수
    int nextFloor;

    //HashSet<int>[] callRequstFloor = new HashSet<int>[(int)MOVE_STATE.end];


    int[] callRequstFloor = new int[(int)MOVE_STATE.end];

    int requestFloor;

    public GameObject[] listFloor;
    public GameObject up, down;
    public GameObject car;

    public TextMeshPro textNo;
    public TextMeshPro textPassenger;
    public TextMeshPro textDoor;


    int elno;
    float preUpdateTime = 0;
    float coolTime = 0;
   

    float currentMoveSpeed;


    MOVE_STATE recv_action;

  
    delegate void ElevatorAction();

    ElevatorAction[] elevatorAction = new ElevatorAction[(int)Event.End];
    public List<ElevatorPassenger> listPassenger = new List<ElevatorPassenger>();


    Fsm<Event, State> fsm = new Fsm<Event, State>();

    Event nextEvent = Event.None;
    float nextTransitionTime =0;


    State reqState = State.Ready;
    int reqfloor = -1;
    float reqTime = 0;


    public int GetNo()
    {
        return elno;
    }


    public MOVE_STATE GetMoveState()
    {
        return moveDirState;
    }

    public Fsm<Event, State> GetFsm()
    {
        return fsm;
    }



    public override void InitializeAgent()
    {
        if (building == null)
            building = FindObjectOfType<Building>();

        base.InitializeAgent();



        for (int i=0; i<callRequstFloor.Length;++i)
        {
            callRequstFloor[i] = -1;
        }
        textPassenger.text = listPassenger.Count.ToString();

        InitFsmFunc();
       
    }

    public void InitFsmFunc()
    {
        fsm.AddStateTransition(State.Ready, Event.Call, State.Accelate);
        fsm.AddStateTransition(State.Ready, Event.Arrived, State.DoorOpening);
        fsm.AddStateTransition(State.Ready, Event.DoorOpenRequest, State.DoorOpening);

        fsm.AddStateTransition(State.Accelate, Event.AccelateEnd, State.NormalMove);
        fsm.AddStateTransition(State.Accelate, Event.DecelerateStart, State.NormalMove);
        fsm.AddStateTransition(State.Accelate, Event.Arrived, State.MoveStop);

        fsm.AddStateTransition(State.NormalMove, Event.DecelerateStart, State.Decelerate);
        fsm.AddStateTransition(State.NormalMove, Event.Arrived, State.MoveStop);

        fsm.AddStateTransition(State.Decelerate, Event.Arrived, State.MoveStop);

        fsm.AddStateTransition(State.MoveStop, Event.DoorOpenRequest, State.DoorOpening);
        fsm.AddStateTransition(State.MoveStop, Event.EmptyPassenger, State.Ready);
        fsm.AddStateTransition(State.MoveStop, Event.DoorCloseEnd, State.Accelate);

        fsm.AddStateTransition(State.DoorOpening, Event.DoorOpenEnd, State.DoorOpened);
        fsm.AddStateTransition(State.DoorOpened, Event.DoorCloseStart, State.DoorClosing);
        fsm.AddStateTransition(State.DoorClosing, Event.DoorCloseEnd, State.Accelate);
        fsm.AddStateTransition(State.DoorClosing, Event.EmptyPassenger, State.Ready);

        elevatorAction[(int)State.Ready] = Ready; //문닫고 멈춰있는 상태
        elevatorAction[(int)State.NormalMove] = NormalMove;   //위아래 어느쪽이든 정상적인 이동상태
        elevatorAction[(int)State.Decelerate] = Decelerate;   //다음층에 멈추기 위한 감속상태
        elevatorAction[(int)State.MoveStop] = MoveStop;
        elevatorAction[(int)State.DoorOpening] = DoorOpening;   //문열는중
        elevatorAction[(int)State.DoorOpened] = DoorOpened;  //문열린 상태에서 승객내리고 타고
        elevatorAction[(int)State.DoorClosing] = DoorClosing;  //문닫히는 동안.
        elevatorAction[(int)State.Accelate] = Accelate;  //이동에 대한 가속상태
        elevatorAction[(int)State.Turn] = Turn;

        fsm.SetCurrentState(State.Ready);


    }


    public void Init()
    {

        while (listPassenger.Count > 0)
        {
            var p = listPassenger[0];
            p.Dispose();
            listPassenger.RemoveAt(0);

        }

        textPassenger.text = listPassenger.Count.ToString();

        for (int i=0;i< floorBtnflag.Length;++i)
        {
            floorBtnflag[i] = false;
            listFloor[i].GetComponent<VerticalLine>().SetDestResquest(false);
        }


        currentMoveSpeed=0;
        recv_action = MOVE_STATE.Stop;
        SetDirction(recv_action);
    
        nextEvent = Event.None;
        nextTransitionTime = 0;
        fsm.SetCurrentState(State.Ready);


        reqState = State.Ready;
        reqfloor = -1;
        reqTime = 0;


        SetPosFloor(Random.Range(0, ElevatorAcademy.floors));

    }

    public void InitFloor(int no, int floors)
    {
        if (resFloor == null)
            resFloor = (GameObject)Resources.Load("Elevator/vertical_line");

        textNo.text = no.ToString();
        elno = no;



        SetPosFloor(Random.Range(0, floors));

        if (listFloor != null)
        {
            if (listFloor.Length == floors)
                return;

            for (int f = floors; f < listFloor.Length; ++f)
            {
                Destroy(listFloor[f]);
            }

            GameObject[] temp = new GameObject[floors];

            for (int f = 0; f < floors; ++f)
            {
                GameObject of;

                if (f >= listFloor.Length || !listFloor[f])
                    of = (GameObject)Instantiate(resFloor, transform);
                else
                    of = listFloor[f];

                of.transform.position = transform.position + (Vector3.up * f * ElevatorAcademy.height);
                temp[f] = of;
            }

            listFloor = temp;
        }
        else
        {
            listFloor = new GameObject[floors];

            for (int f = 0; f < floors; ++f)
            {
                GameObject of = (GameObject)Instantiate(resFloor, transform);
                of.transform.position = transform.position + (Vector3.up * f * ElevatorAcademy.height);
                listFloor[f] = of;

            }
        }

        floorBtnflag = new bool[floors];
    }


    public override void AgentReset()
    {
        base.AgentReset();

    }

    public override void CollectObservations()
    {
        /*
        AddVectorObs(building.GetRestPassenger());  //남은 승객수

        for(int i=0;i<ElevatorAcademy.floors;++i)
        {
            var f= building.GetFloor(i);
            AddVectorObs(f.GetPassengerCount());

            if (f.IsCallRequest(MOVE_STATE.Down))
                AddVectorObs(1);
            else
                AddVectorObs(0);

            if (f.IsCallRequest(MOVE_STATE.Up))
                AddVectorObs(1);
            else
                AddVectorObs(0);
        }


        int floor, nextfloor;
        GetFloor(out floor, out nextfloor);

        AddVectorObs(floor);   ///헌재 층수
        AddVectorObs(nextfloor);   ///다음층수
        AddVectorObs((int)GetMoveState());  ///이동방향
        AddVectorObs((int)fsm.GetCurrentState());
        AddVectorObs(listPassenger.Count);


        for(int i = 0;i<floorBtnflag.Length;++i)
        {
            if(floorBtnflag[i])
                AddVectorObs(1);
            else
                AddVectorObs(0);
        }
        */

        foreach(var el in building.listElve)
        {
            //if(el != this)
            {
                el.MakeObsVector(this);
            }
        }

    }

    public void MakeObsVector(Agent agent)
    {
        agent.AddVectorObs(building.GetRestPassenger());  //남은 승객수

        for (int i = 0; i < ElevatorAcademy.floors; ++i)
        {
            var f = building.GetFloor(i);
            agent.AddVectorObs(f.GetPassengerCount());

            if (f.IsCallRequest(MOVE_STATE.Down))
                agent.AddVectorObs(1);
            else
                agent.AddVectorObs(0);

            if (f.IsCallRequest(MOVE_STATE.Up))
                agent.AddVectorObs(1);
            else
                agent.AddVectorObs(0);
        }


        int floor, nextfloor;
        GetFloor(out floor, out nextfloor);

        agent.AddVectorObs(floor);   ///헌재 층수
        agent.AddVectorObs(nextfloor);   ///다음층수
        agent.AddVectorObs((int)GetMoveState());  ///이동방향
        agent.AddVectorObs((int)fsm.GetCurrentState());
        agent.AddVectorObs(listPassenger.Count);


        for (int i = 0; i < floorBtnflag.Length; ++i)
        {
            if (floorBtnflag[i])
                agent.AddVectorObs(1);
            else
                agent.AddVectorObs(0);
        }
    }


    public void AgentThreeAction(float[] vectorAction, string textAction)
    {
        AddReward(-0.01f);
        recv_action = (MOVE_STATE)Mathf.FloorToInt(vectorAction[0]);


        int floor, nextfloor;
        GetFloor(out floor, out nextfloor);

        var f = building.GetFloor(nextfloor);

        switch (recv_action)
        {
            case MOVE_STATE.Stop:


                {


                    if (fsm.GetCurrentState() == State.NormalMove)
                    {

                        //if(listPassenger.Count==0&&f.listPassenger.Count>0)
                        //{
                        //    AddReward(0.02f);
                        //}

                    }

                    else if (fsm.GetCurrentState() == State.Ready)
                    {

                        while (true && f.listPassenger.Count > 0)
                        {
                            MOVE_STATE dir = (MOVE_STATE)Random.Range((int)MOVE_STATE.Down, (int)MOVE_STATE.end);

                            if (f.IsCallRequest(dir))
                            {

                                fsm.StateTransition(Event.DoorOpenRequest);
                                SetDirction(dir);
                                return;
                            }
                        }

                    }
                }

                fsm.StateTransition(Event.DecelerateStart);

                return;
                break;

            case MOVE_STATE.Down:

                if (floor == 0)
                {
                    // AddReward(-0.01f);
                    return;
                }


                if (GetMoveState() != recv_action)
                {

                    if (listPassenger.Count > 0)
                    {
                        if (currentMoveSpeed > 0.0f)  //이동
                        {
                            AddReward(-0.005f);
                            return;
                        }
                    }
                    else if (currentMoveSpeed > 0.0f)
                    {

                        fsm.StateTransition(Event.DecelerateStart);
                        return;
                    }
                }

                if (!floorBtnflag[nextfloor] && f.listPassenger.Count == 0)
                    AddReward(0.02f);
                else if (f.IsCallRequest(recv_action))
                {
                    AddReward(-0.02f);
                }

                {

                    SetDirction(recv_action);
                    fsm.StateTransition(Event.Call);
                }

                break;

            case MOVE_STATE.Up:


                if (floor == (ElevatorAcademy.floors - 1))
                {
                    // AddReward(-0.01f);
                    return;
                }


                if (GetMoveState() != recv_action)
                {

                    if (listPassenger.Count > 0)
                    {
                        if (currentMoveSpeed > 0.0f)  //이동
                        {
                            AddReward(-0.005f);
                            return;
                        }
                    }
                    else if (currentMoveSpeed > 0.0f)
                    {
                        fsm.StateTransition(Event.DecelerateStart);
                        return;
                    }
                }


                {


                    if (!floorBtnflag[nextfloor] && f.listPassenger.Count == 0)
                        AddReward(0.02f);
                    else if (f.IsCallRequest(recv_action))
                    {
                        AddReward(-0.2f);
                    }

               
                }

                SetDirction(recv_action);
                fsm.StateTransition(Event.Call);
                break;
        }

    }

    public void AgentStopFloor(float[] vectorAction, string textAction)
    {
        int floor = (int)vectorAction[0];


        requestFloor = floor ;   ///층을 배열 인덱스로 접근하기 때문에 -1를 해준다..

        int currentFloor = (int)GetFloor();

        if (fsm.GetCurrentState() == State.Ready)
        {
            if(requestFloor == currentFloor)
            {
                fsm.StateTransition(Event.DoorOpenRequest);
                requestFloor = -1;
                return;
            }


            if (requestFloor > currentFloor)
                SetDirction(MOVE_STATE.Up);
            else
                SetDirction(MOVE_STATE.Down);

            fsm.StateTransition(Event.Call);

        }


    }


    // to be implemented by the developer
    public override void AgentAction(float[] vectorAction, string textAction)
    {

        if(ElevatorAcademy.actionTofloor>0)
            AgentStopFloor(vectorAction, textAction);
        else
            AgentThreeAction(vectorAction, textAction);

    }


    public void SetDirction(MOVE_STATE dir)
    {
        up.SetActive(false);
        down.SetActive(false);

        moveDirState = dir;

        if (dir == MOVE_STATE.Down)
        {
            down.SetActive(true);

        }
        else if (dir == MOVE_STATE.Up)
        {
            up.SetActive(true);
        }

    }

    public void SetLeaveFloor(int floor, bool bOn = true)
    {
        if (listFloor == null || floor >= listFloor.Length)
        {
            Debug.LogError(string.Format("No:{0} SetLeaveFloor VerticalLine{1} Error", textNo.text, floor));
            return;
        }

        listFloor[floor].GetComponent<VerticalLine>().SetDestResquest(bOn);

    }

    public void SetCallFloor(int floor, bool bOn = true)
    {
        if (listFloor == null || floor >= listFloor.Length)
        {
            Debug.LogError(string.Format("No:{0} SetCallFloor VerticalLine{0} Error", textNo.text, floor));
            return;
        }

        listFloor[floor].GetComponent<VerticalLine>().SetDestResquest(bOn);
    }

    public void SetPosFloor(int floor)
    {
        car.transform.position = transform.position + (Vector3.up * floor * ElevatorAcademy.height);
        currentFloor = floor;
    }



    public void UpdatePos()
    {
        if (preUpdateTime == 0)
        {
            preUpdateTime = Time.fixedTime;
            return;
        }

        float delta = Time.fixedTime - preUpdateTime;

        preUpdateTime = Time.fixedTime;

        if (car == null)
        {
            int fssf = 0;
            return;
        }

        if (coolTime > Time.fixedTime)
            return;

        Vector3 movePos = car.transform.position + Vector3.up * moveDir[(int)moveDirState] * currentMoveSpeed * delta;
        car.transform.position = movePos;

        if (car.transform.position.y >= listFloor[listFloor.Length - 1].transform.position.y)
        {
            car.transform.position = listFloor[listFloor.Length - 1].transform.position;
            SetDirction(MOVE_STATE.Down);
           // coolTime = preUpdateTime + ElevatorAcademy.turn;

        }
        else if (car.transform.position.y <= listFloor[0].transform.position.y)
        {
            car.transform.position = listFloor[0].transform.position;
            SetDirction(MOVE_STATE.Up);
            //coolTime = preUpdateTime + ElevatorAcademy.turn;
        }

        currentFloor = (car.transform.localPosition.y / ElevatorAcademy.height)+0.01f;
        CheckFloor();

    }

    public void GetFloor(out int floor, out int nextfloor)
    {
        floor = -1;
        nextfloor = -1;


        switch ((MOVE_STATE)moveDirState)
        {
            case MOVE_STATE.Up:
                floor = (int)currentFloor;
                nextfloor = Mathf.RoundToInt(currentFloor);
                break;

            case MOVE_STATE.Stop:
                floor = (int)currentFloor;
                nextfloor = floor;
                break;

            case MOVE_STATE.Down:
                floor = Mathf.CeilToInt(currentFloor);
                nextfloor = Mathf.RoundToInt(currentFloor);
                break;
        }

       
    }

    public void CheckFloor()
    {
        int floor = -1, nextfloor = -1;

        GetFloor(out floor,out nextfloor);

        if (floor == nextfloor)
            return;

        var fl = building.GetFloor(nextfloor);

        //if (brain.brainType == BrainType.Heuristic
        //           || brain.brainType == BrainType.Player)
        //{

        //    if (!floorBtnflag[nextfloor])
        //    {
        //        RequstAction(nextfloor);
        //    }
        //    else if (fsm.GetCurrentState() != State.Decelerate)
        //    {
        //        fsm.StateTransition(Event.DecelerateStart);
        //    }

        //    else if (callRequstFloor[(int)moveDirState] == nextfloor)
        //    {
        //        fsm.StateTransition(Event.DecelerateStart);
        //        return;
        //    }


        //    if (listPassenger.Count > 0)
        //        return;


        //    bool find = false;
        //    if (moveDirState == MOVE_STATE.Up)
        //    {
        //        find = callRequstFloor[(int)MOVE_STATE.Down] == nextfloor;
        //    }
        //    else
        //    {
        //        find = callRequstFloor[(int)MOVE_STATE.Up] == nextfloor;
        //    }


        //    if (find)
        //    {
        //        fsm.StateTransition(Event.DecelerateStart);
        //    }

        //    return;
        //}



        MOVE_STATE mvstate;
        if (nextfloor == 0)
        {
            mvstate = MOVE_STATE.Up;
        }
        else if (nextfloor >= listFloor.Length - 1)
        {
            mvstate = MOVE_STATE.Down;
        }
        else
        {
            mvstate = moveDirState;
        }


        if (floorBtnflag[nextfloor]|| requestFloor == nextfloor)
        {
            if (fsm.GetCurrentState() != State.Decelerate)
            {
                fsm.StateTransition(Event.DecelerateStart);
            }

            if (requestFloor == nextfloor && fl.IsCallRequest(mvstate))
            {
                AddReward(0.03f);
            }

            return;
        }


       
        if (fl.IsCallRequest(mvstate))
        {
            RequstAction(nextfloor);
        }
        

    }

    public float GetFloor()
    {
        return currentFloor;
    }

    public void UpdateAction()
    {
        if(CheckStateDelay()&& elevatorAction[(int)fsm.GetCurrentState()] != null)
          elevatorAction[(int)fsm.GetCurrentState()]();

        UpdatePos();

        //CheckFloorButton();
    }

    public void SetTransitionDelay(Event evt,float delay = 0.0f,bool bAdd = false)
    {
        if (bAdd)
            nextTransitionTime += delay;
        else
            nextTransitionTime = delay;

        nextEvent = evt;
    }

    public bool CheckStateDelay()
    {
        nextTransitionTime -= Time.fixedDeltaTime;

        if (nextTransitionTime > 0)
            return false;


        if (nextEvent == Event.None || nextTransitionTime==0)
            return true;


        nextTransitionTime = 0;
        fsm.StateTransition(nextEvent);

        nextEvent = Event.None;

        return true;
    }

    public void Ready()
    {
        //아무것도 안하고 대기..
        SetDirction(MOVE_STATE.Stop);
        currentMoveSpeed = 0;

        SetTransitionDelay(Event.End, 0.5f);
        RequstAction((int)GetFloor());

       

    }

    public void Accelate()
    {
        //정상속도로 되기 위해서 가속상태..

        currentMoveSpeed += Time.fixedDeltaTime * ElevatorAcademy.acelerate;

        if (currentMoveSpeed < ElevatorAcademy.speed)
            return;

        currentMoveSpeed = ElevatorAcademy.speed;

        fsm.StateTransition(Event.AccelateEnd);
    }

    public void NormalMove()
    {
        CheckFloor();
    }

    public void Decelerate()
    {
        int nextfloor;

        if(GetMoveState() == MOVE_STATE.Up)
        {
            nextfloor = (int)(currentFloor) + 1;

            if (nextfloor >= listFloor.Length)
                nextfloor = listFloor.Length - 1;
        }
        else
        {
            nextfloor = (int)currentFloor;
        }
        

        float dist = listFloor[nextfloor].transform.position.y - car.transform.position.y;

        if(Mathf.Abs(dist)<= currentMoveSpeed*Time.fixedDeltaTime|| Mathf.Abs(dist)<0.09f)
        {
            car.transform.position = new Vector3(car.transform.position.x, listFloor[nextfloor].transform.position.y, car.transform.position.z);

            currentFloor = (car.transform.localPosition.y / ElevatorAcademy.height);
            fsm.StateTransition(Event.Arrived);
           
            currentMoveSpeed = 0;
            return;
        }

        if (currentMoveSpeed < 0.65)
            return;

        currentMoveSpeed -= Time.fixedDeltaTime * ElevatorAcademy.decelerate;

    }

    public void MoveStop()
    {
        int floor = (int)GetFloor();

        var f = building.GetFloor(floor);


        if (floorBtnflag[floor] ||f.IsCallRequest(GetMoveState()))
        {
            fsm.StateTransition(Event.DoorOpenRequest);
        }
        else if(listPassenger.Count==0&&f.listPassenger.Count>0)
        {

            if(GetMoveState() == MOVE_STATE.Down)
            {
                SetDirction(MOVE_STATE.Up);
            }
            else
            {
                SetDirction(MOVE_STATE.Down);
            }

            fsm.StateTransition(Event.DoorOpenRequest);     
        }
        else if(listPassenger.Count == 0&& f.listPassenger.Count==0)
        {
            fsm.StateTransition(Event.EmptyPassenger);
        }
        else
        {
            fsm.StateTransition(Event.DoorCloseEnd);
        }

        SetFloorButton(floor, false);


        if (requestFloor == floor)
            requestFloor = -1;

    }

    public void DoorOpening()
    {
 
        SetTransitionDelay(Event.DoorOpenEnd, ElevatorAcademy.open);
        textDoor.gameObject.SetActive(!textDoor.gameObject.activeSelf);

       
    }

    public void DoorOpened()
    {
        ///승객 내림처리
     
        float boardingDelay =0;
        int idx = 0;

        int stayfloor = (int)GetFloor();
        switch ((MOVE_STATE)moveDirState)
        {
            case MOVE_STATE.Up:
               
                stayfloor = Mathf.RoundToInt(currentFloor);
                break;

            case MOVE_STATE.Stop:
                stayfloor = (int)currentFloor;
                break;

            case MOVE_STATE.Down:
                stayfloor = Mathf.RoundToInt(currentFloor);
                break;
        }


        while (idx < listPassenger.Count)
        {
            var p = listPassenger[idx];
            if (p.destFloor == stayfloor)
            {
                listPassenger.RemoveAt(idx);
                boardingDelay += Random.Range(0.6f, 1.0f);

                float refTime = Mathf.Abs((p.startFloor - p.destFloor) * (ElevatorAcademy.height) / ElevatorAcademy.speed/2f);
                AddReward(refTime / (Time.fixedTime - p.timeWaiting));
                AddReward(0.0001f);

                p.Dispose();
                building.AddDestPassenger();
            }
            else
            {
                ++idx;
            }
        }

        SetTransitionDelay(Event.DoorCloseStart, boardingDelay);
        


    }

    public void DoorClosing()
    {
        if (listPassenger.Count > 0)
        {
            ///승객이 있을 경우는 다시 이동을 하도록 셋팅해준다..
            SetTransitionDelay(Event.DoorCloseEnd, 1.0f);
        }
        else
        {
            ///승객이 없을 경우는 일단 해당층에 서 대기한다.
            SetTransitionDelay(Event.EmptyPassenger, 1.0f);                                                                            
        }

        textDoor.gameObject.SetActive(true);

        textPassenger.text = listPassenger.Count.ToString();
    }

    public void Turn()
    {

    }

    public IEnumerator SetTranstionEvent(Event e, float waitTime)
    {
        yield return new WaitForSeconds(waitTime);

        fsm.StateTransition(e);

        yield break;
    }

    public void SetFloorButton(int floor,bool bOn)
    {
        listFloor[floor].GetComponent<VerticalLine>().SetDestResquest(bOn);

        floorBtnflag[floor] = bOn;

        if(moveDirState == (int)MOVE_STATE.Stop&& bOn)
        {
            if(floor > currentFloor)
            {
                SetDirction(MOVE_STATE.Up);
            }
            else if (floor < currentFloor)
            {
                SetDirction(MOVE_STATE.Down);
            }          
        }

    }


    public void CheckFloorButton()
    {
        if(moveDirState == MOVE_STATE.Stop)
        {

        }
    }

    public bool EnterPassenger(ElevatorPassenger p)
    {
        SetFloorButton(p.destFloor, true);

        listPassenger.Add(p);

        return true;
    }

    public float GetFloorDist(int floor,MOVE_STATE dir)
    {

        float dist = listFloor[floor].transform.position.y - car.transform.position.y;

        if (moveDirState == (int)MOVE_STATE.Stop)
            return Mathf.Abs(dist);

        if (moveDirState == MOVE_STATE.Up)
        {
            if(dist > 0)
            {
                return Mathf.Abs(dist);
            }
            else
            {
                dist = Mathf.Abs(listFloor[ElevatorAcademy.floors - 1].transform.position.y - car.transform.position.y);
                dist += Mathf.Abs(listFloor[ElevatorAcademy.floors - 1].transform.position.y - listFloor[floor].transform.position.y);
                return dist;

            }

        }


        if (dist < 0)
        {
            return Mathf.Abs(dist);
        }
        else                                   ///내려올경우..
        {
            dist = Mathf.Abs(listFloor[0].transform.position.y - car.transform.position.y);
            dist += Mathf.Abs(listFloor[0].transform.position.y - listFloor[floor].transform.position.y);

        }

        return dist;
        
    }

    public bool SetCallRequest(int floor,MOVE_STATE dir)
    {
       if(fsm.GetCurrentState() == State.Ready)
        {
            if(floor == GetFloor())
            {
                fsm.StateTransition(Event.DoorOpenRequest);
                SetDirction(dir);
                return true;
            }


            if(floor>GetFloor())
            {
               
                SetDirction(MOVE_STATE.Up);
            }
            else
            {
                SetDirction(MOVE_STATE.Down);
            }
            fsm.StateTransition(Event.Call);

            return true;

        }

        return false;

    }

    public bool IsEnterableState()
    {
        if (fsm.GetCurrentState() != State.DoorOpened
            ||listPassenger.Count>=ElevatorAcademy.capacity)
            return false;

        return true;
    }

    public bool AddPassenger(ElevatorPassenger p)
    {
        if (listPassenger.Count >= ElevatorAcademy.capacity)
            return false;

        if(GetFloor() == requestFloor)
            AddReward(0.01f);

        if (GetMoveState() == MOVE_STATE.Up && p.destFloor > GetFloor())
        {
            listPassenger.Add(p);
            SetTransitionDelay(Event.DoorCloseStart, Random.Range(0.6f, 1.0f),true);         
            SetFloorButton(p.destFloor, true);

            //AddReward(5f/ (Time.fixedTime - p.timeWaiting));
            //p.timeWaiting = Time.fixedTime;
        }
        else if (GetMoveState() == MOVE_STATE.Down && p.destFloor < GetFloor())
        {
            listPassenger.Add(p);
            SetTransitionDelay(Event.DoorCloseStart, Random.Range(0.6f, 1.0f), true);
            SetFloorButton(p.destFloor, true);

            //AddReward(1f / (Time.fixedTime - p.timeWaiting));
            //p.timeWaiting = Time.fixedTime;
        }
        else  
        {
            return false;
        }

      
        return true;
    }

    public void RequstAction(int floor)
    {

        //if(brain.brainType == BrainType.Heuristic
        //    || brain.brainType == BrainType.Player)
        //{

        //    if(fsm.GetCurrentState() == State.NormalMove)
        //    {
        //        var f = building.GetFloor(floor);

        //        if (floor == 0)
        //        {
        //            if (f.IsCallRequest(MOVE_STATE.Up))
        //            {
        //                fsm.StateTransition(Event.DecelerateStart);
        //                return;
        //            }
        //        }
        //        else if (floor == ElevatorAcademy.floors - 1)
        //        {
        //            if (f.IsCallRequest(MOVE_STATE.Down))
        //            {
        //                fsm.StateTransition(Event.DecelerateStart);
        //                return;
        //            }
        //        }
        //        else
        //        {
        //            if (f.IsCallRequest(GetMoveState()))
        //            {
        //                fsm.StateTransition(Event.DecelerateStart);
        //                return;
        //            }
        //        }

        //        if (listPassenger.Count==0&&building.IsNoCallRequest())
        //        {
        //            fsm.StateTransition(Event.DecelerateStart);
        //            return;
        //        }

        //    }

        //    return;
        //}


        if(fsm.GetCurrentState() == State.Ready)
        {
            if (Time.fixedTime - reqTime < 0.1f)
                return;
        }
        else
        {
            if (reqState == fsm.GetCurrentState() && reqfloor == floor)
            {

                if(reqfloor == floor)

                return;
            }

        }


        RequestDecision();

        reqState = fsm.GetCurrentState();
        reqfloor = floor;
        reqTime = Time.fixedTime;

    }

   
}
