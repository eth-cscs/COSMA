/*****************************************************************************    
  TimerCpp : A C++ tool to measure times and extract statistics
  Copyright (C) 2014  Manos Tsardoulias

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*****************************************************************************/

#include "timer.h"

void function_B();
void function_C();
void function_D();

void function_A(void)
{
  TimerCpp::start("func_a", "main_timer");
  usleep(100000);
  for(unsigned int i = 0 ; i < 15 ; i++)
  {
    function_B();
  }
  function_D();
  TimerCpp::tic("func_a");
}

void function_B(void)
{
  TimerCpp::start("func_b", "func_a");
  usleep(1000);
  for(unsigned int i = 0 ; i < 15 ; i++)
  {
    function_C();
  }
  TimerCpp::tic("func_b");
}

void function_C(void)
{
  TimerCpp::start("func_c", "func_b");
  usleep(5000);
  TimerCpp::tic("func_c");
}

void function_D(void)
{
  TimerCpp::start("func_d", "func_a");
  usleep(3000);
  TimerCpp::tic("func_d");
}

int main(int argc, char ** argv)
{
  TimerCpp::start("main_timer");
  for(unsigned int i = 0 ; i < 5 ; i++)
  {
    function_A();
  }
  TimerCpp::tic("main_timer");
  
  TimerCpp::printAllMeans();
  
  TimerCpp::printAllMeansTree();
  return 0;
}
