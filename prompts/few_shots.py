from utils.xml_parser import XMLParser

tool_parser = XMLParser(fields=["think", ("tool", "answer")])
result_parser = XMLParser(fields=["result"])

SEARCH_FEW_SHOT = [
    [
        {'role': 'user', 'content': 'Question: Which of the following is an example of mechanical weathering?\n\nChoices:\nA. A rock turns red when iron minerals are exposed to oxygen.\nB. A rock breaks apart after water freezes and expands in a crack.\nC. A rock dissolves as acidic rainwater flows over its surface.\nD. A rock changes composition when exposed to extreme heat.'},
        {'role': 'assistant', 'content': tool_parser.format(
            think="I need to understand what mechanical weathering is and which of these options matches that phenomenon. Let me search for information about mechanical weathering.",
            tool='{"name": "search", "args": {"query": "what is mechanical weathering examples", "num_results": 3}}'
        )},
        {'role': 'user', 'content': result_parser.format(
            result="• Mechanical Weathering - Definition, Processes and Types\n  Mechanical weathering, also known as physical weathering, is the process by which rocks are broken down into smaller pieces by external forces. Examples include frost wedging (when water freezes and expands in cracks), thermal expansion/contraction, and pressure release.\n\n• Mechanical and Chemical Weathering - Earth Science\n  Mechanical weathering breaks rocks into smaller pieces without changing their composition. Ice wedging occurs when water gets into cracks, freezes, and expands, breaking the rock apart. Other forms include plant root growth, animal activity, and abrasion.\n\n• Mechanical Weathering: Definition, Types, Examples - Geology Science\n  Types of mechanical weathering include frost wedging, salt crystallization, thermal expansion, plant activity, and animal activity. The most common mechanical weathering process in cold climates is frost wedging, where water enters rock cracks, freezes, and expands."
        )},
        {'role': 'assistant', 'content': tool_parser.format(
            think="Based on the search results, mechanical weathering is the process by which rocks are broken down into smaller pieces by external forces, without changing their chemical composition. Some examples include:\n\n1. Frost wedging - when water gets into cracks, freezes, and expands, breaking the rock apart\n2. Thermal expansion/contraction\n3. Plant root growth\n4. Animal activity\n5. Abrasion\n\nLooking at the answer choices:\nA. A rock turns red when iron minerals are exposed to oxygen - This is chemical weathering, as it involves a chemical change (oxidation).\nB. A rock breaks apart after water freezes and expands in a crack - This is frost wedging, a form of mechanical weathering.\nC. A rock dissolves as acidic rainwater flows over its surface - This is chemical weathering (dissolution).\nD. A rock changes composition when exposed to extreme heat - This is a chemical/thermal change, not mechanical weathering.\n\nThe answer is B, as it's the only option that describes mechanical weathering.",
            answer="B"
        )}
    ]
]

PLAN_FEW_SHOT = [
    [
        {'role': 'user', 'content': 'Task: I need to create a complex Simulink electrical power system model. Can you help me plan this step by step?'},
        {'role': 'assistant', 'content': tool_parser.format(
            think="This is a complex task that would benefit from a structured approach. I should break it down into manageable steps and create a plan.",
            tool='{"name": "plan", "args": {"plan_list": ["Analyze the system requirements and identify key components", "Search for appropriate Simulink blocks for power system modeling", "Design the overall system architecture", "Implement the basic bus structure", "Add generation sources and loads", "Connect all components with proper electrical connections", "Test and validate the model"]}}'
        )},
        {'role': 'user', 'content': result_parser.format(
            result="Plan created with 7 steps:\n1. Analyze the system requirements and identify key components\n2. Search for appropriate Simulink blocks for power system modeling\n3. Design the overall system architecture\n4. Implement the basic bus structure\n5. Add generation sources and loads\n6. Connect all components with proper electrical connections\n7. Test and validate the model"
        )},
        {'role': 'assistant', 'content': tool_parser.format(
            think="Great! I've created a structured plan for building the Simulink power system. Now let me start executing the first step.",
            answer="Continue"
        )},
        {'role': 'user', 'content': result_parser.format(
            result="Step 1 completed:\nTask: Analyze the system requirements and identify key components\n<result>\nFor a complex electrical power system in Simulink, the key components typically include:\n\n1. Buses - Connection points for electrical components\n2. Generators - Power sources (synchronous machines, renewable sources)\n3. Transformers - Voltage level conversion\n4. Transmission lines - Power transfer between buses\n5. Loads - Power consumers (static, dynamic, motor loads)\n6. Protection systems - Circuit breakers, relays\n7. Control systems - Voltage and frequency regulation\n8. Measurement blocks - Monitoring and data collection\n\nThe system should follow standard power system conventions with proper grounding, three-phase modeling, and realistic electrical parameters.\n</result>\n\nRemaining steps: 6"
        )},
        {'role': 'assistant', 'content': tool_parser.format(
            think="Excellent! The first step provided a comprehensive analysis of power system components. Based on this analysis, I should refine the plan to be more specific about the implementation details. I'll reference the completed analysis step and revise the remaining steps.",
            tool='{"name": "plan", "args": {"plan_list": ["Search for Three-Phase V-I Measurement, Synchronous Machine, Three-Phase Transformer blocks", "Design 4-bus system architecture with generator, transformer, and load placements", "Create buses using Three-Phase V-I Measurement blocks", "Add Synchronous Machine generator and Three-Phase Series RLC loads", "Implement Three-Phase PI Section Lines and Three-Phase Transformer connections", "Configure electrical parameters and test the complete model"]}}'
        )},
        {'role': 'user', 'content': result_parser.format(
            result="Plan updated. 1 steps completed, 6 remaining:\n✓ 1. Analyze the system requirements and identify key components\n○ 2. Search for Three-Phase V-I Measurement, Synchronous Machine, Three-Phase Transformer blocks\n○ 3. Design 4-bus system architecture with generator, transformer, and load placements\n○ 4. Create buses using Three-Phase V-I Measurement blocks\n○ 5. Add Synchronous Machine generator and Three-Phase Series RLC loads\n○ 6. Implement Three-Phase PI Section Lines and Three-Phase Transformer connections\n○ 7. Configure electrical parameters and test the complete model"
        )},
        {'role': 'assistant', 'content': tool_parser.format(
            think="Perfect! The plan has been updated with more specific implementation steps. Now let me continue with the next step.",
            answer="Continue"
        )}
    ],
    [
        {'role': 'user', 'content': 'I started working on a research project but I think my approach needs revision. Can you help me create a better plan?'},
        {'role': 'assistant', 'content': tool_parser.format(
            think="The user wants to revise their research approach. I should create a comprehensive research plan that can be iteratively refined.",
            tool='{"name": "plan", "args": {"plan_list": ["Define research question and objectives clearly", "Conduct literature review to understand current state", "Identify research methodology and approach", "Design experiments or studies", "Collect and analyze data", "Draw conclusions and write up results"]}}'
        )},
        {'role': 'user', 'content': result_parser.format(
            result="Plan created with 6 steps:\n1. Define research question and objectives clearly\n2. Conduct literature review to understand current state\n3. Identify research methodology and approach\n4. Design experiments or studies\n5. Collect and analyze data\n6. Draw conclusions and write up results"
        )},
        {'role': 'assistant', 'content': tool_parser.format(
            think="I've created a basic research plan. Let me start with the first step to help define the research question and objectives.",
            answer="Continue"
        )},
        {'role': 'user', 'content': result_parser.format(
            result="Step 1 completed:\nTask: Define research question and objectives clearly\n<result>\nResearch Planning Framework:\n\n1. Research Question: What specific problem are you trying to solve?\n2. Objectives: What are the measurable goals?\n3. Scope: What are the boundaries and limitations?\n4. Success Criteria: How will you know if you've succeeded?\n5. Timeline: What are the key milestones?\n\nPlease provide more details about your research area and what you're trying to achieve so I can help you formulate specific questions and objectives.\n</result>\n\nRemaining steps: 5"
        )},
        {'role': 'assistant', 'content': tool_parser.format(
            think="The first step completed successfully and provided a framework for research planning. The user should now provide more specific details about their research area. Let me finish this plan for now since we need user input to proceed meaningfully.",
            answer="Finish"
        )},
        {'role': 'user', 'content': result_parser.format(
            result="Plan finished. Completed 1/6 steps."
        )},
        {'role': 'assistant', 'content': 'I\'ve helped you start with a research planning framework. As the first step indicated, I\'ll need more specific details about your research area and objectives to provide more targeted guidance. What field are you researching, and what specific problem are you trying to solve?'}
    ]
]

CREATE_SYSTEM_FEW_SHOT = [
    [
        {
            "role": "user",
            "content": (
                "Question: Your task is to generate a Python dictionary representation of a Simulink system "
                "based on the following description: This electrical system contains 2 buses, 1 load, "
                "plus 1 external grid. Bus1 routes power to Bus2. ExternalGrid1 feeds Bus2. "
                "Load1 feeds Bus1."
            )
        },
        {
            "role": "assistant",
            "content": tool_parser.format(
                think="I need to search the relevant blocks in Simulink to create a Python dictionary for this system.",
                tool='{"name": "search_blocks", "args": {"query_list": ["bus", "ExternalGrid", "load"]}'
            )
        },
        {
            "role": "user",
            "content": result_parser.format(
                result=(
                    "bus: Three-Phase V-I Measurement\n"
                    "ExternalGrid: Three-Phase Source\n"
                    "load: Three-Phase Series RLC Load"
                )
            )
        },
        {
            "role": "assistant",
            "content": tool_parser.format(
                think=(
                    "Based on the search results, I can create a Python dictionary representation of this two-bus system. "
                    "The blocks include:\n\n"
                    "- Bus: Three-Phase V-I Measurement\n"
                    "- External Grid: Three-Phase Source\n"
                    "- Load: Three-Phase Series RLC Load\n\n"
                    "Now, I will create the dictionary."
                ),
                answer="""```python
system_dict = {
    "Blocks": {
        "Bus1" : {
            "Type": "Three-Phase V-I Measurement"
        },
        "Bus2" : {
            "Type": "Three-Phase V-I Measurement"
        },
        "ExternalGrid1" : {
            "Type": "Three-Phase Source"
        },
        "Load1" : {
            "Type": "Three-Phase Series RLC Load"
        }
    },
    "Connections": [
        ["Bus1/(A2,B2,C2)", "Bus2/(A1,B1,C1)"],
        ["ExternalGrid1/(A1,B1,C1)", "Bus2/(A2,B2,C2)"],
        ["Load1/(A1,B1,C1)", "Bus1/(A2,B2,C2)"]
    ]
}
```""")}
    ],
    [
        {'role': 'user', 'content': 'Question: Your task is to generate a Python dictionary representation of a Simulink system based on the following description: This electrical system contains 4 bus(es), 1 transformer(s), 2 load(s), plus 1 external grid(s). Bus1 routes power to Line1. Bus4 connects to Line2, and Transformer1. Line1 is connected to Bus4. Line2 is connected to Bus2. Transformer1 connects to Bus3. Load1 feeds Bus1. Load2 feeds Bus3. ExternalGrid1 feeds Bus2.'},
        {'role': 'assistant', 'content': tool_parser.format(
            think="I need to search the relevant blocks in Simulink to create a Python dictionary for the four-bus system.",
            tool='{"name": "search_blocks", "args": {"query_list": ["bus", "generator", "power line", "transformer", "ExternalGrid"], "num_results": 1}}'
        )},
        {'role': 'user', 'content': result_parser.format(
            result="bus: Three-Phase V-I Measurement\n  generator: Synchronous Machine\n  power line: Three-Phase PI Section Line\n  transformer: Three-Phase Transformer\n external grid: Three-Phase Source"
        )},
        {'role': 'assistant', 'content': tool_parser.format(
            think="Based on the search results, I can create a Python dictionary representation of the Simulink system. The blocks include:\n\n- Bus: Three-Phase V-I Measurement\n- Generator: Synchronous Machine\n- Power Line: Three-Phase PI Section Line\n- Transformer: Three-Phase Transformer\n\nNow, I will create the dictionary.",
            answer="""```python
system_dict = {
    "Blocks": [
        "Bus1",
        "Bus2",
        "Bus3",
        "Bus4",
        "Line1",
        "Line2",
        "Transformer1",
        "Load1",
        "Load2",
        "ExternalGrid1"
    ],
    "Connections": [
        {
            "Source": "Bus1",
            "Destination": "Line1"
        },
        {
            "Source": "Line1",
            "Destination": "Bus4"
        },
        {
            "Source": "Bus4",
            "Destination": "Line2"
        },
        {
            "Source": "Line2",
            "Destination": "Bus2"
        },
        {
            "Source": "Bus4",
            "Destination": "Transformer1"
        },
        {
            "Source": "Transformer1",
            "Destination": "Bus3"
        },
        {
            "Source": "ExternalGrid1",
            "Destination": "Bus2"
        },
        {
            "Source": "Load1",
            "Destination": "Bus1"
        },
        {
            "Source": "Load2",
            "Destination": "Bus3"
        }
    ]
}
```""")}
    ]
]