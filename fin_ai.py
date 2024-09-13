import os 
import json
import requests
from datetime import date
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import streamlit as st

st.set_page_config(page_title="Financials Agent AI", page_icon="📊", layout="wide")

if "sectors_api_key" not in st.session_state:
    st.session_state["sectors_api_key"] = ""
if "groq_api_key" not in st.session_state:
    st.session_state["groq_api_key"] = ""

with st.sidebar:
    SECTORS_API_KEY = st.text_input(
        "Sectors API Key", key="sectors_api_key", type="password"
    )
    GROQ_API_KEY = st.text_input("Groq API Key", key="groq_api_key", type="password")
    button = st.button("Set API Keys")
def retrieve_from_endpoint(url: str) -> dict:
    headers = {"Authorization": SECTORS_API_KEY}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    return json.dumps(data)

@tool
def get_company_report(stock: str, sections: str) -> str:
    """
    Get data about the company report with stock symbol parameters and the desired section from IDX,
    refer to url: https://sectors.app/api for full documentation 
    :param stock: receive a stock symbol from the company which consists of 4 letters.
    :param sections: This parameter accepts sections which are overview, valuation, 
    future, peers, financials, dividend, management, ownership.
    - Overview: overview explains the overview report from the company, 
    such as industry, sub-sector, email, phone number, market cap, market cap rank, etc.
    - Valuation: This section explain the company's management, capital structure, 
    future earnings and the market value of its assets.
    - Future: Explain the future of the company such as growth forecasts and the like
    - Peers: Explains company peer comparison or perhaps sector comparison
    - Financials: Explain the financial condition of the company such as data revenue, earnings, and so on
    - Dividend: Explains the dividend history and stability of the company
    - Management: Explain the management of the company
    - Ownership: Explain the share ownership of the company
    """
    valid_sections = ["overview", "valuation", "future", "peers", "financials", 
                      "dividend", "management", "ownership"]
    assert sections in valid_sections, "Please specify a section from the documentation"
    assert (len(stock) == 4), "Symbol saham harus 4 digit, e.g BBRI"

    url = f"https://api.sectors.app/v1/company/report/{stock}/?sections={sections}"
    return retrieve_from_endpoint(url)

@tool
def get_top_companies_by_tx_volume(
    start_date: str, end_date: str, top_n: int = 5
) -> str:
    """
    Get the most traded company or top companies by transaction volume and most traded stock by volume from IDX.
    :param start_date: start date is the desired initial date to search for data on volume and
    price transactions from a company
    :param end_date: end date is the desired date to find volume and price transaction data for a company's stock
    :param top_n: is a parameter for finding data from n number of companies with the highest value
    with the default value is 5
    This function will produce data from companies with the most traded volume transactions. 
    and will give results in the form of the total volume, symbol, company name and 
    average price of each symbol in a period from start date to end date. 
    If the start date and end date are the same then it will produce the total volume, symbol, company name and average price.
    """
    assert top_n > 0, "Please enter a valid value (cannot be minus and '0')"
    
    url = f"https://api.sectors.app/v1/most-traded/?start={start_date}&end={end_date}&n_stock={top_n}"
    
    data = retrieve_from_endpoint(url)

    data_dict = json.loads(data)
    
    # Aggregate volume across all dates
    aggregated_data = {}
    for date_data in data_dict.values():
        for record in date_data:
            symbol = record['symbol']
            volume = record['volume']
            price = record['price']
            company_name = record['company_name']
            if symbol in aggregated_data:
                aggregated_data[symbol]['volume'] += volume
                # Update the total price and count for calculating the average later
                aggregated_data[symbol]['avg_price'] += price
                aggregated_data[symbol]['count'] += 1
            else:
                aggregated_data[symbol] = {
                'symbol': symbol,
                'volume': volume,
                'avg_price': price,
                'count': 1,
                'company_name': company_name
            }
    
    # Calculate the average price for each symbol
    for symbol, data in aggregated_data.items():
        aggregated_data[symbol]['avg_price'] = data['avg_price'] / data['count']

    # Sort by total volume and get top N
    sorted_data = sorted(aggregated_data.values(), key=lambda x: x['volume'], reverse=True)[:top_n]
    
    return json.dumps(sorted_data)

@tool
def get_daily_tx(stock: str) -> str:
    """
    Get detail of stock prices or close price to see uptrend or downtrend
    and you can also see the number of company market caps per day.
    :param stock: receive a stock symbol from the company which consists of only 4 letters.
    :param start_date: receive the desired date to get the start date of the company's daily transaction data
    :param end_date: receive end_date or the end date of the desired daily transaction data
    This tool will provide symbol, date, close price, volume, and market cap.
    """
    url = f"https://api.sectors.app/v1/daily/{stock}/"

    return retrieve_from_endpoint(url)

@tool
def get_company_performance_ipo(stock: str) -> str:
    """
    Get company's performance since IPO from IDX.
    """
    url = f"https://api.sectors.app/v1/listing-performance/{stock}/"

    return retrieve_from_endpoint(url)

@tool
def get_companies_by_subsector(sub_sector: str):
    """
    Get companies by sub sector
    Valid subsector : alternative-energy, apparel-luxury-goods, automobiles-components,
                          banks, basic-materials, consumer-services, financing-service,
                          food-beverage, food-staples-retailing, healthcare-equipment-providers,
                          heavy-constructions-civil-engineering, holding-investment-companies,
                          household-goods, industrial-goods, industrial-services, insurance,
                          investment-service, leisure-goods, logistics-deliveries, media-entertainment,
                          multi-sector-holdings, nondurable-household-products, oil-gas-coal,
                          pharmaceuticals-health-care-research, properties-real-estate,
                          retailing, software-it-services, technology-hardware-equipment,
                          telecommunication, tobacco, transportation, transportation-infrastructure,
                          utilities.
    """
    url = f"https://api.sectors.app/v1/companies/?sub_sector={sub_sector}"

    return retrieve_from_endpoint(url)

@tool
def get_subsector_report(sector: str, section: str) -> str:
    """
    Get sub sector report from IDX.
    """
    url = f"https://api.sectors.app/v1/subsector/report/{sector}/?sections={section}"

    return retrieve_from_endpoint(url)

def get_today_date() -> str:
    """
    Get today's date
    """

    today = date.today()
    return today.strftime("%Y-%m-%d")

tools = [
    get_company_report,
    get_top_companies_by_tx_volume,
    get_daily_tx,
    get_company_performance_ipo,
    get_companies_by_subsector
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """"Answer the following queries, being as factual and analytical 
             as you can. If you need the start and end dates but they are not 
             explicitly provided, infer from the query. Whenever you return a 
             list of names, return also the corresponding values for each name.
             And use the tools.
             For comparison, ensure you provide the correct answer based on the specific query. 
             Accurately retrieve and compare data when required, and choose the right entity or value.

             You are a highly skilled financial agent and please make sure the answer is right.
             
             If the volume was about a single day, the start and end parameter 
             should be the same. Answer based on the question. 

             if asked to display results from one date, always display one date 
             and do not add any other dates than requested.
            
             Find and return data for only the single closest available date to the specified date, 
             whether it's before or after. Do not return more than one date, 
             and make sure to choose the closest one chronologically.

             If an empty result or empty list is obtained, 
             then try moving to the next date and continue doing so until an answer is found and
             and if you have the answer, then use that date to answer the question, 
             make sure to choose the closest one chronologically.
             
             If there is an answer with a lot of numbers, use dots or commas to make it easier to read.

             Always answer in markdown and markdown table if necessary and possible.

             Today date is
             """
             + get_today_date()
        ),
        ("human", "{input}"),
        # msg containing previous agent tool invocations 
        # and corresponding tool outputs
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

llm = ChatGroq(
    temperature=0,
    model_name="llama3-groq-70b-8192-tool-use-preview",
    groq_api_key=GROQ_API_KEY,
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, 
                               tools=tools, 
                               verbose=True
                               )

st.title("🤖Financials Agent AI")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

if prompt := st.chat_input():
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🧠 thinking..."):
            st_callback = StreamlitCallbackHandler(st.container())
            try:
                response = agent_executor.invoke({"input": prompt}, callbacks=st_callback)

                answer = response.get('output', 'No response received.')
                st.session_state.messages.append({'role': 'assistant', 'content': answer})

                #st.write(response)
                st.success(response['output'])
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # HTTP 429 Too Many Requests
                    st.error("We've reached our API usage limit. Please wait a few minutes and try again.")
                else:
                    st.error(f"Oops! Something went wrong: {e}. Please try again later.")
            except requests.exceptions.RequestException as e:
                st.error(f"Network issue detected: {e}. Please check your connection or try again later.")
            except Exception as e:
                st.error(f"Unexpected error: {e}. Please try again or contact support if the issue persists.")


