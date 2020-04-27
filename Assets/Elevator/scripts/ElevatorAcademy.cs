using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;
public class ElevatorAcademy : Academy
{

    // Use this for initialization

    Building building;

    public static int elevatorCount;
    public static int floors;
    public static int passenger;
    public static float height;
    public static float speed;
    public static float decelerate;
    public static float acelerate;
    public static float open;
    public static float close;
    public static float turn;
    public static int capacity;
    public static int actionTofloor;



    void Start ()
    {
       
    }
	
	// Update is called once per frame
	void Update ()
    {
		
	}


    public override void InitializeAcademy()
    {
        building = FindObjectOfType<Building>();


        floors = (int)resetParameters["floor"];
        elevatorCount = (int)resetParameters["elevators"];
        passenger = (int)resetParameters["passenger"];
        height= resetParameters["height"]; 
        speed = resetParameters["speed"]; 
        decelerate = resetParameters["decelerate"];
        acelerate = resetParameters["acelerate"];
        open = resetParameters["open"];
        close = resetParameters["close"];
        turn = resetParameters["turn"];
        capacity = (int)resetParameters["capacity"];
        actionTofloor = (int)resetParameters["actionTofloor"];



        building.InitEnv();
    }

    public override void AcademyReset()
    {
        building.InitEnv();
    }

    public override void AcademyStep()
    {
        building.UpdateEnv();
    }
}
