<p align="center">
  <img src="https://github.com/khu-capstone-design-case/case-backend/assets/82189072/38682ca6-d5e2-4e32-a382-2d47d6154469.png" width="320" height="320"/>
</p>

'CASE'(**Ca**ll Recording **S**ummary & **E**motion Analysis) 프로젝트의 AI 서버에 사용되는 레포지토리입니다.
## 👉 연계 Frontend Repo
> https://github.com/khu-capstone-design-case/case-front

## 👉 연계 Backend Repo
> https://github.com/khu-capstone-design-case/case-backend

---
## 프로젝트 소개
'CASE'는 통화 녹음 요약 및 감정(뉘앙스) 분석 웹 서비스입니다.

## 개발 동기 및 개발 목적
SK텔레콤이 발표한 인공지능 통화 녹음 서비스 에이닷(A.)의 월간 활성 이용자 수가 100만 명을 넘어서는 등 음성녹음 및 기록에 대한 관심이 매우 뜨겁다. 정식 버전으로 출시된 지 한 달 만이다.

이러한 현상은 통화 녹음 기능을 원해왔던 아이폰 이용자들, 그리고 통화 내용을 텍스트로 요약해 주는 기능 덕분인 것으로 보인다.


하지만 “티켓값”을 “튀김 값”으로 잘못 인식하는 등 정확성이 아직 미흡하며, 제공하는 기능이 지나치게 다양해 이용하기 복잡하다는 평가가 잇따랐다. 

또한 에이닷은 SKT 사용자만 이용 가능하고, 에이닷을 사용하려면 기본 전화 앱을 에이닷으로 설정해야 한다. 따라서 이용자는 기본 전화, T 전화, 그리고 에이닷까지 총 3개의 앱 중 선택해야 하는 상황이 발생한다.

따라서 통신사, 기기 및 환경 설정에 구애받지 않는 어플리케이션 Case를 제안한다. Case는 파일을 업로드하는 방식으로 통화 녹음을 텍스트로 변환 및 요약해 주며, 나아가 문장 속에 담긴 감정까지 파악해주는 기능을 탑재하고 있다.

Case는 Pretrained 모델 및 Fine-tuning된 모델을 활용하여 STT (Speech To Text)기능을 제공하고, 음성 seperation을 일대 다로 확장하여 대화를 일대일통화로만 제한하지 않는다. 

또한 변환된 문장들을 선택한 뒤, 문장에 담긴 감정이 어떠한지 판단해 주는 기능을 도입하였다. 

단순히 대화를 저장하는 것에서 나아가 연인이나 친구, 지인과의 대화에서 뉘앙스 속에 담긴 감정을 파악하는 데 도움을 줄 수 있도록 하였다. 

중요하거나, 추억하고 싶은 대화를 텍스트로 저장 및 요약하고 싶은 이용자를 앱의 주 타겟층으로 한다. STT의 오류율은 CER 10%이하를 목표로 한다.

## 개발 환경 및 사용 기술
- FrontEnd : React, TypeScript, Vite
- BackEnd : Spring Boot, MySQL, MongoDB, Docker
- AI: PyTorch, FastApi, Naver Clova AI, ChatGPT
- 공통 : Git, Postman, AWS EC2
- 협업툴 : Notion, GitHub, Figma, Zoom

## 페이지별 기능
### 회원가입, 로그인 페이지
<img src="https://github.com/khu-capstone-design-case/case-backend/assets/82189072/ec66e42f-56fe-4d27-a9ec-4a9f63effb67.jpeg" width="320" height="800"/> |<img src="https://github.com/khu-capstone-design-case/case-backend/assets/82189072/f5a66103-9eb4-4ffb-b15c-1dfa0fa284fb.jpeg" width="320" height="800"/>
--- | --- |

새로운 사용자가 서비스에 가입하는 회원 가입 과정과 가입한 사용자가 자신의 계정에 접속하는 로그인 과정이다. 

회원가입시 아이디(이메일), 비밀번호, 비밀번호 재입력, 이름을 입력받으며, 아이디는 중복확인을 진행한다. 

이름은 추후 통화 녹음 스크립트를 조회할 때 통화자로써 사용된다.

### 통화 녹음 파일 업로드 페이지
<img src="https://github.com/khu-capstone-design-case/case-backend/assets/82189072/3db81617-8765-4ad1-8deb-801daafa489b.jpeg" width="320" height="800"/> |<img src="https://github.com/khu-capstone-design-case/case-backend/assets/82189072/d8b5a4a9-87cd-4bd8-9153-ad039e6a7572.jpeg" width="320" height="800"/>
--- | --- |

사용자가 통화 녹음 파일을 선택하여 등록하는 기능이다. 

첫 번째 화면에서 대화 추가를 누르면 통화 녹음 업로드 화면이 나온다. 

통화 녹음 업로드 화면에서 마이크 버튼을 클릭하면 자신의 디바이스 에서 통화 녹음 파일을 업로드할 수 있다.

### 통화 녹음 파일 분석 페이지
<img src="https://github.com/khu-capstone-design-case/case-backend/assets/82189072/c2ac9787-e221-4688-a65f-a4455866c620.jpeg" width="320" height="480"/> |<img src="https://github.com/khu-capstone-design-case/case-backend/assets/82189072/b3718232-16e3-4c2b-8d0a-9a0ec18ac1c0.jpeg" width="320" height="480"/> |<img src="https://github.com/khu-capstone-design-case/case-backend/assets/82189072/90ac498c-d47e-49d8-a996-be9b23e6dcb6.jpeg" width="320" height="480"/> |<img src="https://github.com/khu-capstone-design-case/case-backend/assets/82189072/cafb46bf-65f3-4b4c-ae8e-54a5cba4b7eb.jpeg" width="320" height="480"/>  
--- | --- | --- | --- |

통화 녹음 파일을 업로드 한 후 통화 녹음의 정보를 추가적으로 입력하는 데, 추가 정보는 통화 상대 이름, 총 통화자 수이다. 

추가 정보를 입력한 후 완료 버튼을 누르게 되면 통화 녹음 분석을 진행하게 된다. 

통화 녹음 분석 과정은 파일 변환, 음성 분리, 음성 인식, 감정 분석, 내용 요약 순으로 진행된다.

### 통화 녹음 파일 분석 결과 확인 페이지
<img src="https://github.com/khu-capstone-design-case/case-backend/assets/82189072/9066f558-8ef0-413f-b774-190c7f1b2ef0.jpeg" width="320" height="480"/> |<img src="https://github.com/khu-capstone-design-case/case-backend/assets/82189072/f4c109f0-6a64-47a1-8ad5-a4ed01486e95.jpeg" width="320" height="480"/> |<img src="https://github.com/khu-capstone-design-case/case-backend/assets/82189072/0b019d3b-6a47-4f2d-a888-211ce12a33cb.jpeg" width="320" height="480"/> |<img src="https://github.com/khu-capstone-design-case/case-backend/assets/82189072/9aa6909b-b9f9-4377-bce2-1e32fd2ecb3d.jpeg" width="320" height="480"/>  
--- | --- | --- | --- |

통화 녹음 파일 분석이 종료되면 분석이 완료된 통화 녹음 목록 화면으로 바뀌게 된다. 

통화 녹음 목록에서 해당하는 통화 녹음을 선택하게 되면 통화 녹음 스크립트를 볼 수 있고, 통화 녹음 오디오를 들을 수 있다. 

이때 스크립트 마다의 뉘앙스를 분석한 결과도 나타나게 되는데, 긍정이 높은 스크립트는 초록색으로, 중립이 높은 스크립트는 회색으로, 부정이 높은 스크립트는 빨간색으로 표시된다. 

또한 통화 녹음의 분석 결과에서 사용자와 통화 상대의 대화가 반대로 나타나는 경우, 화자 변경 버튼을 통해 화자를 반대로 변경할 수 있다. 

화면 하단의 “대화 선택하기”를 누르면 각각의 대화 옆에 표시된 점이 나오는데 이 점들을 누르고 화면 하단의 “뉘앙스 판단하기”를 누르면 선택한 스크립트들의 뉘앙스 분석 결과를 확인할 수 있다.

### 통화 녹음 상대방, 목록 삭제 페이지
<img src="https://github.com/khu-capstone-design-case/case-backend/assets/82189072/7e205ec4-b731-4735-aa76-a74d1a699499.jpeg" width="320" height="480"/> |<img src="https://github.com/khu-capstone-design-case/case-backend/assets/82189072/35a0673f-4116-4d71-bbb1-72e60c140e0c.jpeg" width="320" height="480"/> |<img src="https://github.com/khu-capstone-design-case/case-backend/assets/82189072/70352290-3902-4a8b-8b8b-b4df400453ae.jpeg" width="320" height="480"/>
--- | --- | --- |


화면의 우 상단에는 휴지통 모양이 있는데 이것을 클릭하면 음성 파일을 삭제할 수 있다. 

통화 녹음 1개를 삭제할 수도 있고, 통화 상대방을 삭제하면 해당하는 통화 녹음 목록 전체가 삭제된다. 

하지만 음성 파일 분석이 진행되고 있는 도중에는 파일을 삭제할 수 없다.

## 기대 효과
많은 통화 녹음을 가지고 있는 사람에게 녹음 오디오 파일 하나 하나 들으면서 내용 파악을 할 필요 없이 

스크립트와 요약을 통해 빠르게 원하는 통화 녹음을 찾고 들을 수 있게 하여 통화 녹음을 찾는 불필요한 시간을 줄일 수 있을 것으로 기대할 수 있고, 

자동으로 통화 녹음을 스크립트로 변환해주기 때문에 통화 녹음 내용을 파악하거나 사용해야할 때도 용이할 것이다.

또한 스크립트 각각의 감정 분석과 여러 스크립트를 선택하여 나오는 새로운 맥락과 뉘앙스를 보다 객관적으로 확인할 수 있어

연인과 친구, 지인간의 대화에서 대화의 뉘앙스 파악, 더 나아가 해당 발언의 의도도 파악할 수 있어 더욱 더 원활한 소통이 가능해질 것으로 기대하고 있다.
