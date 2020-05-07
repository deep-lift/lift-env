using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;



public class Building : MonoBehaviour
{
   
    static ElevatorAcademy academy;
    public static Brain elevatorBrain;

    // Use this for initialization

    static GameObject resElevator;
    static GameObject resfloor;
    static float simulation_interval = 3f;

    public List<ElevatorAgent> listElve = new List<ElevatorAgent>();
    List<Buildfloor> listFloor = new List<Buildfloor>();

    ElevatorAgent[,] callReqReserveCar;


    static int episodeTotalPassenger ;

    static GameObjPool s_GameObjPool;

    int currentPassenger;
    int restPassenger;
    int destPassenger;
    int addPassenger;


    public AnimationCurve simuPassenger;

    

    float simulattion_time = 0;

    float startTime = 0;

    int success = 0;
    int fail = 0;

    struct PassengerSpawn
    {
        public int step;
        public int floor;
        public int passengercount;
    }


    int scenario_index = 0;
    List<PassengerSpawn> passenger_scenario = new List<PassengerSpawn>();


    private void Start()
    {
        academy.AgentSetStatus += SetStatus;
    }

    // Update is called once per frame
    void Update () {
		
	}


    public void InitEnv()
    {
        if (s_GameObjPool == null)
            s_GameObjPool = new GameObjPool();

        if (resElevator == null)
            resElevator = (GameObject)Resources.Load("Elevator/elevator_unit");

        if(resfloor == null )
            resfloor = (GameObject)Resources.Load("Elevator/build_floor");

        if (academy ==null)
            academy = FindObjectOfType<ElevatorAcademy>();


        callReqReserveCar = new ElevatorAgent[ElevatorAcademy.floors, 2];

        ElevatorPassenger.InitPooler();


        //if (elevatorBrain == null)
        //    elevatorBrain = academy.transform.Find("ElevatorBrain").GetComponent<Brain>();



        restPassenger = ElevatorAcademy.passenger;
        destPassenger = 0;
        simulattion_time = 0;
        addPassenger = 0;
        scenario_index = 0;


        int dist = 4;
        int rest = ElevatorAcademy.elevatorCount% 2;
        int mok = ElevatorAcademy.elevatorCount / 2;

        Vector3 startPos = transform.position;
        if (rest<0.5f)
        {
            mok -= 1;
            startPos = transform.position - (Vector3.right * dist * mok)- (Vector3.right * (dist/2)); 
        }
        else
        {
            startPos = transform.position - (Vector3.right * dist * mok);
        }

        startPos += Vector3.back;


        for (int i = 0; i< ElevatorAcademy.elevatorCount; ++i)
        {

            if(i< listElve.Count)
            {
                listElve[i].Init();
                continue;
            }

            GameObject ele = (GameObject)Instantiate(resElevator, this.transform);
            ele.transform.position = startPos + (Vector3.right * dist * i);

            var agent = ele.GetComponent<ElevatorAgent>();
            listElve.Add(agent);
            //agent.GiveBrain(elevatorBrain);
            agent.InitFloor(i, ElevatorAcademy.floors);
            //agent.agentParameters.agentCameras[0] = GameObject.Find("agent_cam").GetComponent<Camera>();
            agent.AgentReset();

        }

        for (int i = 0; i < ElevatorAcademy.floors; ++i)
        {

            if (i < listFloor.Count)
            {
                listFloor[i].Init();
                continue;
            }


            GameObject fl = (GameObject)Instantiate(resfloor, this.transform);
            fl.transform.position = transform.position + (Vector3.up * ElevatorAcademy.height * i);
            fl.GetComponent<Buildfloor>().SetFloor(i,this);
            listFloor.Add(fl.GetComponent<Buildfloor>());

        }

        startTime = Time.fixedTime;


    }


    public  void SetStatus(bool academyMaxStep, bool academyDone, int academyStepCounter)
    {
        if(academyMaxStep)
        {
            foreach (var el in listElve)
            {
                el.SetReward(-(GetRestPassenger() * 0.1f));
                el.Done();
                fail += 1;
                academy.Done();
                el.RequestDecision();
            }
        }

    }


    public void UpdateEnv()
    {

        SimulationFloorPassenger();

        SimulationEnterElevator();

        UpdatePos();

        if (IsDone())
        {
          
            foreach (var el in listElve)
            {
                el.SetReward(1f);
                el.Done();
                el.RequestDecision();
            }

            success += 1;

            academy.Done();
            return;
        }
/*
        if (academy.IsDone())
        {
            foreach (var el in listElve)
            {
                el.SetReward(-(GetRestPassenger()*0.1f));
                el.Done();
                fail += 1;
                academy.Done();
                el.RequestDecision();
            }

        }
*/
    }

    public void UpdatePos()
    {
        foreach (var e in listElve)
        {
            e.UpdateAction();
        }
    }


    public void SimulationFloorPassenger()
    {
        if (academy.GetEpisodeCount() == 1)
        {
            PassengerRandomSpawn();
        }
        else
        {
            PassengerScenarioSpawn();
        }


    }

    public void PassengerScenarioSpawn()
    {


      


        while (scenario_index < passenger_scenario.Count)
        {
            if(passenger_scenario[scenario_index].step>=academy.GetStepCount())
            {
                listFloor[passenger_scenario[scenario_index].floor].GetComponent<Buildfloor>().AddPassenger(passenger_scenario[scenario_index].passengercount);
                addPassenger += passenger_scenario[scenario_index].passengercount;
                restPassenger -= passenger_scenario[scenario_index].passengercount;
                ++scenario_index;
            }
            else
            {
                break;
            }
        }

    }

    public void PassengerRandomSpawn()
    {
        if (simulattion_time > Time.fixedTime)
            return;

        if (currentPassenger > episodeTotalPassenger * 0.3)
            return;

        int newPassenger = Random.Range(0, restPassenger + 1);

        int[] floorPassenger = new int[listFloor.Count];



        floorPassenger[0] = Random.Range(0, (int)(newPassenger * 0.8f));

        int rest = newPassenger - floorPassenger[0];


        while (rest > 0)
        {
            int floor = Random.Range(1, listFloor.Count);
            int passenger = Random.Range(1, rest + 1);
            rest -= passenger;
            floorPassenger[floor] = passenger;
        }


        for (int i = 0; i < listFloor.Count; ++i)
        {
            if (floorPassenger[i] > 0)
            {
                listFloor[i].GetComponent<Buildfloor>().AddPassenger(floorPassenger[i]);
                addPassenger += floorPassenger[i];
                restPassenger -= floorPassenger[i];

                if(academy.GetEpisodeCount()==1)
                {
                    PassengerSpawn spawn;
                    spawn.floor = i;
                    spawn.step = academy.GetStepCount();
                    spawn.passengercount = floorPassenger[i];

                    passenger_scenario.Add(spawn);
                }
                 
            }


        }

        simulattion_time = Time.fixedTime + 5f;


    }


    public void SimulationEnterElevator()
    {

        for (int i = 0; i < listFloor.Count; ++i)
        {
            foreach(var el in listElve)
            {
                listFloor[i].EnterElevator(el);
            }
        }

    }


    public void CallRequest(int floor, MOVE_STATE dir)
    {
      
        //switch(elevatorBrain.brainType)
        //{
        //    case BrainType.Player:
        //    case BrainType.Heuristic:
        //        SearchRuleBaseNearstElevator(floor, dir);
        //        break;

        //    case BrainType.External:
        //    case BrainType.Internal:
        //        break;


        //    default:
        //        break;
        //}
        
    }

    public void ProcRuleBaseCallRequest()
    {
        foreach(var f in listFloor)
        {
            for(int i = (int)MOVE_STATE.Down;i< (int)MOVE_STATE.end;++i)
            {
                SearchRuleBaseNearstElevator(f.GetFloorNo(), (MOVE_STATE)i);
            }
        }

    }

    public int SearchRuleBaseNearstElevator(int floor,MOVE_STATE dir)
    {

        float min = 1000000f;
        float dist = 0;
        int buttonDir = 0;

        if(dir != MOVE_STATE.Down)
        {
            buttonDir = 1;
        }
       

        foreach(var e in listElve)
        {
            dist = e.GetFloorDist(floor, dir);

            if (dist < min)
            {
                callReqReserveCar[floor, buttonDir] = e;
                min = dist;
            }
        }

        if (callReqReserveCar[floor, buttonDir] != null)
        {
            var el = callReqReserveCar[floor, buttonDir];
            el.SetCallRequest(floor, dir);
            return el.GetNo();
        }

        return -1;
    }

    public Buildfloor GetFloor(int floor)
    {

        return listFloor[floor];

    }

    public MOVE_STATE GetAction(int floor,ElevatorAgent el)
    {
        return MOVE_STATE.Stop;
    }

    public void AddDestPassenger(int add =1)
    {
        destPassenger += add;
    }

    public bool IsNoCallRequest()
    {
        foreach (var f in listFloor)
        {
            if (!f.IsNoCall())
                return false;
        }

        return true;
    }

    public bool IsDone()
    {
        if (restPassenger > 0)
            return false;


        foreach (var el in listElve)
        {
            if (el.listPassenger.Count > 0)
                return false;
        }

        return IsNoCallRequest();

    }

    public int GetRestPassenger()
    {
        return ElevatorAcademy.passenger-destPassenger;
    }

    private void OnGUI()
    {
        if (academy == null)
            return;

        GUI.TextArea(new Rect(10, 10, 200,25),
            string.Format("EP:{0}-Step:{1} Suc:{2}", academy.GetEpisodeCount(), academy.GetStepCount(), success));

        GUI.TextArea(new Rect(10, 40, 200, 25),
          string.Format("Passenger:{0}/{1}", destPassenger, ElevatorAcademy.passenger));
    }



}
