using System;
using System.Collections.Generic;
using System.Text;


public class FiniteState< EVENT,STATE >
{	
	public FiniteState( STATE dwStateID ){ m_dwStateID = dwStateID; }

    public STATE GetStateID() { return m_dwStateID; }
    public void AddTransition(EVENT inputEvent,STATE outputStateID)
    {

        if (mapTransition.ContainsKey(inputEvent))
            mapTransition[inputEvent] = outputStateID;
        else
            mapTransition.Add(inputEvent, outputStateID);
        
    }

    public void DeleteTransition(EVENT inputEvent)
    {
        mapTransition.Remove(inputEvent);
    }

    public STATE OutputState(EVENT inputEvent)
    {
        STATE outState;

        if (!mapTransition.TryGetValue(inputEvent, out outState))
            return GetStateID();

        return outState;
    }

    public int GetCount() { return mapTransition.Count; }
    private STATE m_dwStateID;
    private Dictionary<EVENT, STATE> mapTransition=new Dictionary<EVENT, STATE>();
	
};

public class Fsm<EVENT, STATE> 
{
    public	Fsm() {}

    public void AddStateTransition(STATE stateID, EVENT inputEvent, STATE outputStateID)
    {
        FiniteState<EVENT, STATE> State;

       if(!mapState.TryGetValue(stateID,out State))
       {
           //  만일 동일한 State가 존재하지 않는다면 새로 생성한다.
           State=new FiniteState<EVENT, STATE>(stateID);
           mapState.Add(stateID, State);
       }
       //  상태 전이 정보를 추가한다.
       State.AddTransition(inputEvent, outputStateID); 
    }

	public void DeleteTransition( STATE stateID, EVENT inputEvent )
    {
        FiniteState<EVENT, STATE> State;

        if (mapState.TryGetValue(stateID, out State))
        {
            State.DeleteTransition(inputEvent);

            if (State.GetCount() == 0)
                mapState.Remove(stateID);

        }

    }

    public STATE GetOutputState(EVENT inputEvent)
    {
        FiniteState<EVENT, STATE> State;

        if (mapState.TryGetValue(GetCurrentState(), out State))
            return State.GetStateID();

        return GetCurrentState();
    }

    public void SetCurrentState(STATE stateID)
    {

        FiniteState<EVENT, STATE> State;

        if (mapState.TryGetValue(stateID, out State))
        {
            m_pCurrState = State;
            stateTransitionTime = UnityEngine.Time.fixedTime;
        }
    }

    public  STATE GetCurrentState()  
    {
        if (m_pCurrState == null)
            return default(STATE);

        return m_pCurrState.GetStateID();
    } 

	public void StateTransition(EVENT nEvent)
    {
      if(m_pCurrState == null)
		return;

	    STATE outputState= m_pCurrState.OutputState(nEvent);
        SetCurrentState(outputState);
    }

    public float GetTransitionTime()
    {
        return stateTransitionTime;
    }

    private Dictionary<STATE, FiniteState<EVENT, STATE>> mapState= new Dictionary<STATE, FiniteState<EVENT, STATE>>();
    private FiniteState<EVENT, STATE> m_pCurrState=null;
    private float stateTransitionTime;



};

