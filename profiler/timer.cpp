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

//~ Static veriable instrantiations
std::map<std::string,double> TimerCpp::times = 
  std::map<std::string,double>();
  
std::map<std::string,double> TimerCpp::max_time = 
  std::map<std::string,double>();
  
std::map<std::string,double> TimerCpp::min_time = 
  std::map<std::string,double>();
  
std::map<std::string,double> TimerCpp::mean_time = 
  std::map<std::string,double>();
  
std::map<std::string,double> TimerCpp::sum_time = 
  std::map<std::string,double>();
  
std::map<std::string,unsigned long> TimerCpp::count =
  std::map<std::string,unsigned long>();

std::map<std::string,std::set<std::string> > TimerCpp::timer_tree =
  std::map<std::string,std::set<std::string> >();

std::string TimerCpp::top_node = "";

struct timeval TimerCpp::msTime = timeval();

//~ ---------------- PRIVATE METHODS ----------------------//

void TimerCpp::printMsInternal(double t)
{
  std::cout << t << " ms";
}
void TimerCpp::printSecInternal(double t)
{
  std::cout << t / 1000.0 << " sec";
}
void TimerCpp::printMinInternal(double t)
{
  std::cout << t / 1000.0 / 60.0 << " minutes";
}
void TimerCpp::printHoursInternal(double t)
{
  std::cout << t / 1000.0 / 60.0 / 60.0 << " hours";
}
void TimerCpp::printLiteralInternal(double t)
{
  int sec = t / 1000.0;
  if(sec >= 1)
  {
    t -= sec * 1000;
    int min = sec / 60.0;
    if(min >= 1)
    {
      sec -= min * 60;
      int hours = min / 60.0;
      if(hours >= 1)
      {
        min -= hours * 60;
        std::cout <<
          min << " hours " <<hours << " minutes "
          << sec << " sec " << t << " ms";
      }
      else
      {// Mins
        std::cout <<
          min << " minutes " << sec << " sec " << t << " ms\n";
      }
    }
    else
    {// Sec
      std::cout << sec << " sec " << t << " ms";
    }
  }
  else
  {// Ms
    std::cout << t << " ms";
  }
}

void TimerCpp::printLiteralMeanInternal(std::string timerId, std::string identation)
{
  MsdIt it = times.find(timerId);
  if(it == times.end())
  {
    std::cout << "Invalid timer id : " << timerId << "\n";
  }
  else
  {
    std::cout << identation << timerId <<
      " [" << times[timerId] << " - " <<
      min_time[timerId] << " , " << mean_time[timerId] << " , " <<
      max_time[timerId] << " - " << count[timerId] << "]";
    std::cout << std::endl;
  }
}

void TimerCpp::printIterativeTree(std::string node, std::string identation)
{
  printLiteralMeanInternal(node,identation);
  for(std::set<std::string>::iterator it = timer_tree[node].begin() ; it !=
    timer_tree[node].end() ; it++)
  {
    printIterativeTree(*it,identation + "|-");
  }
}

//~ ---------------- PUBLIC METHODS ----------------------//

void TimerCpp::start(std::string timerId, std::string father)
{
  MsdIt it = times.find(timerId);
  if(it == times.end())
  {
    timer_tree.insert(std::pair<std::string, std::set<std::string> >
      (timerId,std::set<std::string>()));
    gettimeofday(&msTime, NULL);
    double ms = (double)msTime.tv_sec * 1000 + (double)msTime.tv_usec / 1000 ;
    times.insert(Psd(timerId,ms));
    count.insert(Psul(timerId,0));
    mean_time.insert(Psul(timerId,0));
    sum_time.insert(Psul(timerId,0));
    max_time.insert(Psd(timerId,-1.0));
    min_time.insert(Psd(timerId,1000000000000000.0));
  }
  else
  {
    gettimeofday(&msTime, NULL);
    double ms = (double)msTime.tv_sec * 1000 + (double)msTime.tv_usec / 1000 ;
    it->second = ms;
    mean_time.insert(Psul(timerId,0));
    sum_time.insert(Psul(timerId,0));
    max_time.insert(Psd(timerId,-1.0));
    min_time.insert(Psd(timerId,1000000000000000.0));
  }
  if(father != "")
  {
    timer_tree[father].insert(timerId);
  }
  else
  {
    top_node = timerId;
  }
}

double TimerCpp::stop(std::string timerId)
{
  MsdIt it = times.find(timerId);
  if(it == times.end())
  {
    std::cout << "Invalid timer id : " << timerId << "\n";
    return -1;
  }
  else
  {
    gettimeofday(&msTime , NULL);
    double ms = (double)msTime.tv_sec * 1000 + (double)msTime.tv_usec / 1000 ;
    return ms - it->second;
  }
}

double TimerCpp::mean(std::string timerId)
{
  MsdIt it = times.find(timerId);
  if(it == times.end())
  {
    std::cout << "Invalid timer id : " << timerId << "\n";
    return -1;
  }
  else
  {
    return mean_time[timerId];
  }
}

void TimerCpp::tic(std::string timerId)
{
  MsdIt it = times.find(timerId);
  if(it == times.end())
  {
    std::cout << "Invalid timer id : " << timerId << "\n";
  }
  else
  {
    count[timerId]++;
    gettimeofday(&msTime , NULL);
    double ms = (double)msTime.tv_sec * 1000 + (double)msTime.tv_usec / 1000 ;
    ms -= it->second;
    times[timerId] = ms;
    sum_time[timerId] += ms;
    mean_time[timerId] = sum_time[timerId] / count[timerId];
    if(min_time[timerId] > ms)
      min_time[timerId] = ms;
    if(max_time[timerId] < ms)
      max_time[timerId] = ms;
  }
}

void TimerCpp::printMs(std::string timerId)
{
  std::cout << "Timer " << timerId << " : ";
  printMsInternal(stop(timerId));
  std::cout << std::endl;
}

void TimerCpp::printSec(std::string timerId)
{
  std::cout << "Timer " << timerId << " : ";
  printSecInternal(stop(timerId));
  std::cout << std::endl;
}

void TimerCpp::printMin(std::string timerId)
{
  std::cout << "Timer " << timerId << " : ";
  printMinInternal(stop(timerId));
  std::cout << std::endl;
}

void TimerCpp::printHours(std::string timerId)
{
  std::cout << "Timer " << timerId << " : ";
  printHoursInternal(stop(timerId));
  std::cout << std::endl;
}

void TimerCpp::printLiteral(std::string timerId)
{
  std::cout << "Timer " << timerId << " : ";
  printLiteralInternal(stop(timerId));
  std::cout << std::endl;
}

void TimerCpp::printLiteralMean(std::string timerId)
{
  MsdIt it = times.find(timerId);
  if(it == times.end())
  {
    std::cout << "Invalid timer id : " << timerId;
    std::cout << std::endl;
  }
  else
  {
    std::cout << timerId << " [" << times[timerId] << " - " <<
      min_time[timerId] << " , " << mean_time[timerId] << " , " <<
      max_time[timerId] << " - " << count[timerId] << "]";
    std::cout << std::endl;
  }
}

void TimerCpp::printAll(void)
{
  std::cout << "Timers available : [curr - min , mean , max - tics]\n";
  for(MsdCIt it = times.begin() ; it != times.end() ; it++)
  {
    std::cout << "\t";
    printLiteral(it->first);
    std::cout << "\n";
  }
}

void TimerCpp::printAllMeans(void)
{
  std::cout << "Timers available : [curr - min , mean , max - tics]\n";
  Mfs tms;
  for(MsdCIt it = mean_time.begin() ; it != mean_time.end() ; it++)
  {
    tms.insert(std::pair<float,std::string>(it->second,it->first));
  }
  for(MfsIt it = tms.begin() ; it != tms.end() ; it++)
  {
    std::cout << "\t"; 
    printLiteralMean(it->second);
  }
}

void TimerCpp::printAllMeansTree(void)
{
  std::cout <<
    "Timers available : [curr - min , mean , max - tics]\n";
  Mfs tms;
  for(MsdCIt it = mean_time.begin() ; it != mean_time.end() ; it++)
  {
    tms.insert(std::pair<float,std::string>(it->second,it->first));
  }
  printIterativeTree(top_node,"  ");
}
