% experimentreport.cls
% 实验报告 LaTeX 模板类文件

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{experimentreport}[2026/06/11 Course Experiment Report Class]

\LoadClass[12pt,a4paper]{ctexart}

% 页面设置
\RequirePackage{geometry}
\geometry{
  left=2.6cm,
  right=2.6cm,
  top=2.2cm,
  bottom=2.2cm
}

% 常用宏包
\RequirePackage{array}
\RequirePackage{setspace}
\RequirePackage{tabularx}
\RequirePackage{titlesec}

% 基本格式
\pagestyle{empty}
\setlength{\parindent}{2em}
\setstretch{1.35}
\renewcommand{\arraystretch}{1.45}

\newcommand{\placeholder}[2]{
\begin{center}
\fbox{
\parbox[t][#1][t]{0.9\textwidth}{
#2
}
}
\end{center}
}

\usepackage[normalem]{ulem}
\newcommand{\hintholder}[1]{%
\uline{\hfill\textnormal{\small #1}\hfill}
}

% 标题格式
\titleformat{\section}
  {\zihao{4}\heiti}
  {\chinese{section}、}
  {0em}
  {}

% 信息字段默认值
\newcommand{\coursename}{基础软件开发技术实践}
\newcommand{\reportname}{智能软件工程实验报告}
\newcommand{\experimenttitle}{实验\_\_：\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_}

\newcommand{\studentname}{}
\newcommand{\department}{}
\newcommand{\studentid}{}
\newcommand{\courseteacher}{}
\newcommand{\instructor}{}
\newcommand{\lablocation}{}
\newcommand{\labtime}{}
\newcommand{\performance}{}
\newcommand{\reportscore}{}
\newcommand{\totalscore}{}
\newcommand{\operationscore}{}

% 用户可编辑命令
\newcommand{\setcoursename}[1]{\renewcommand{\coursename}{#1}}
\newcommand{\setreportname}[1]{\renewcommand{\reportname}{#1}}
\newcommand{\setexperimenttitle}[1]{\renewcommand{\experimenttitle}{#1}}

\newcommand{\setstudentname}[1]{\renewcommand{\studentname}{#1}}
\newcommand{\setdepartment}[1]{\renewcommand{\department}{#1}}
\newcommand{\setstudentid}[1]{\renewcommand{\studentid}{#1}}
\newcommand{\setcourseteacher}[1]{\renewcommand{\courseteacher}{#1}}
\newcommand{\setinstructor}[1]{\renewcommand{\instructor}{#1}}
\newcommand{\setlablocation}[1]{\renewcommand{\lablocation}{#1}}
\newcommand{\setlabtime}[1]{\renewcommand{\labtime}{#1}}
\newcommand{\setperformance}[1]{\renewcommand{\performance}{#1}}
\newcommand{\setreportscore}[1]{\renewcommand{\reportscore}{#1}}
\newcommand{\settotalscore}[1]{\renewcommand{\totalscore}{#1}}
\newcommand{\setoperationscore}[1]{\renewcommand{\operationscore}{#1}}

% 封面标题
\newcommand{\makereporttitle}{
\begin{center}
  {\zihao{2}\heiti \coursename}

  \vspace{0.6cm}

  {\zihao{2}\heiti \reportname}

  \vspace{0.6cm}

  {\zihao{3}\heiti \experimenttitle}
\end{center}

\vspace{0.4cm}
}

% 信息表
\newcommand{\makereportinfo}{
\begin{center}
\begin{tabular}{|p{2.8cm}|p{4.2cm}|p{2.8cm}|p{4.2cm}|}
\hline
姓名 & \studentname & 院系 & \department \\ \hline
学号 & \studentid & 任课教师 & \courseteacher \\ \hline
指导教师 & \instructor & 实验地点 & \lablocation \\ \hline
实验时间 & \labtime &  &  \\ \hline
实验得分 &  \reportscore & 研讨得分 & \performance \\ \hline
报告得分 & \operationscore & 实验总分 & \totalscore \\ \hline
\end{tabular}
\end{center}

\vspace{0.8cm}
}

% 完整报告头部
\newcommand{\makereporthead}{
  \makereporttitle
  \makereportinfo
}

% 固定章节命令
\newcommand{\experimentpurpose}[1]{
\section{实验目的}
#1
}

\newcommand{\experimentcontent}[1]{
\section{实验内容}
#1
}

\newcommand{\experimentresult}[1]{
\section{实验结果}
要求：将实验获得的结果进行描述，基本内容包括：

\vspace{0.3cm}

#1
}

\newcommand{\experimentproblems}[1]{
\section{实验中遇到的问题总结}
要求：主要阐述实验过程中遇到的问题如何解决的。

\vspace{0.3cm}

#1
}

\newcommand{\experimentexperience}[1]{
\section{实验体会}
#1
}

% 指导教师评语
\newcommand{\teachercomment}{
\vspace{1cm}

\noindent{\heiti 指导教师评语：}

\vspace{4cm}

\noindent 日期：\hspace{4cm}
}

\endinput