import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import time


class EnergyProfilerClient:
    """
    This class provides methods to interact with the OpenAI API for creating and managing
    an assistant specialized in household energy profiling.
    """

    def __init__(self):
        """
        Initializes the EnergyProfilerClient class by setting the OpenAI API key.
        """
        load_dotenv()
        openai.api_key = os.environ['OPENAI_API_KEY']
        self.client = OpenAI()

    def create_file(self):
        """
        Creates a file in the OpenAI client for processing.

        Returns:
            The created file object.
        """
        with open("data/energy_data.txt", "rb") as file:
            return self.client.files.create(file=file, purpose='assistants')

    def create_assistant(self):
        """
        Creates an assistant specialized in household energy profiling.

        Returns:
            The created assistant object.
        """
        return self.client.beta.assistants.create(
            name="Household Energy Profiler",
            instructions=("You are an experienced Household energy profiler. Your job is to "
                          "analyse, summarise, and provide insights from data and create a consumer profile"),
            tools=[{"type": "code_interpreter"}],
            model="gpt-4"
        )

    def create_thread(self):
        """
        Creates a new thread for communication with the assistant.

        Returns:
            The created thread object.
        """
        return self.client.beta.threads.create()

    def create_message(self, thread_id, file_id):
        """
        Creates a message in a given thread with specific instructions and associated file.

        Args:
            thread_id (str): The ID of the thread.
            file_id (str): The ID of the associated file.

        Returns:
            The created message object.
        """
        return self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=("Create an infographic or a series of visualizations that illustrate an AI-driven "
                     "analysis of household energy consumption. The data, compiled in the 'energy_data.csv' file, "
                     "includes electricity consumption (kWh/h), stock exchange electricity prices (c/kWh), and outdoor "
                     "temperature data (°C) from January 2022 to September 2023. The analysis should focus on a "
                     "fictional customer with an electric-heated detached house under a stock exchange electricity "
                     "contract, utilizing cheaper night-time electricity rates. The visualizations should highlight "
                     "patterns and insights into the customer's electricity usage, such as periods of high consumption, "
                     "efficiency in using night-time rates, and potential areas for energy savings. Additionally, include "
                     "AI-recommended strategies for better utilizing favorable electricity contract terms and adapting to "
                     "demand response. Ensure the visualizations are clear, informative, and effectively convey the intricate "
                     "relationships between consumption, pricing, and external factors like temperature. Note that the data "
                     "is separated with ';' and decimal point can be presented by ','"),
            file_ids=[file_id]
        )

    def run_thread(self, thread_id, assistant_id):
        """
        Executes the assistant in a given thread and monitors its status until completion.

        Args:
            thread_id (str): The ID of the thread.
            assistant_id (str): The ID of the assistant.

        Returns:
            The final run status of the assistant.
        """
        run = openai.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)
        
        while run.status != "completed":
            run = openai.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            time.sleep(10)
            print(run.status)

        return run

    def get_messages(self, thread_id):
        """
        Retrieves all messages from a specified thread.

        Args:
            thread_id (str): The ID of the thread.

        Returns:
            A list of messages from the thread.
        """
        return openai.beta.threads.messages.list(thread_id=thread_id)


if __name__ == "__main__":
    # Example usage
    client = EnergyProfilerClient()
    file = client.create_file()
    assistant = client.create_assistant()
    thread = client.create_thread()
    client.create_message(thread_id=thread.id, file_id=file.id)
    final_run = client.run_thread(thread_id=thread.id, assistant_id=assistant.id)
    messages = client.get_messages(thread_id=thread.id)

    print('Final result:')
    result = messages.data[0]
    print(result)
