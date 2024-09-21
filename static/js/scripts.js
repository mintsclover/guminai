document.addEventListener('DOMContentLoaded', function () {
    // 기존 변수 선언
    const sendButton = document.getElementById('send-button');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const modelSelect = document.getElementById('model-select');
    const modelIntro = document.getElementById('model-intro');

    // 기본값 HTML 설정
    const defaultChatLogoHTML = `
    <div class="chat-logo" id="chat-logo">
        <img src="/static/images/favicon.png" alt="GuminAI 로고">
        <p>안녕하세요! 구미나이입니다. 무엇을 도와드릴까요?</p>
    </div>
    `;
    chatMessages.innerHTML = defaultChatLogoHTML;

    // 봇 아바타 이미지 소스 변수 추가
    let botAvatarSrc = '/static/images/bot_avatar.png';
    updateBotAvatar();

    // 모델 설명 및 봇 아바타 업데이트 함수
    function updateModelIntro() {
        const selectedModel = modelSelect.value;
        if (selectedModel === 'model1') {
            modelIntro.textContent = '환영합니다! 기억 기능이 업데이트 되었습니다!';
        } else if (selectedModel === 'model2') {
            modelIntro.textContent = '환영합니다! 기억 기능이 업데이트 되었습니다!';
        } else {
            modelIntro.textContent = '환영합니다! 기억 기능이 업데이트 되었습니다!';
        }
        // 모델에 따라 봇 아바타 이미지 변경
        updateBotAvatar();
    }

    // 봇 아바타 이미지 업데이트 함수
    function updateBotAvatar() {
        const selectedModel = modelSelect.value;
        if (selectedModel === 'model1') {
            botAvatarSrc = '/static/images/bot_avatar_model1.png';
        } else if (selectedModel === 'model2') {
            botAvatarSrc = '/static/images/bot_avatar_model2.png';
        } else if (selectedModel === 'model3') {
            botAvatarSrc = '/static/images/bot_avatar_model3.png';
        } else {
            botAvatarSrc = '/static/images/bot_avatar.png'; // 기본 아바타
        }
    }

    // 초기 로드 시 모델 소개 업데이트
    updateModelIntro();

    modelSelect.addEventListener('change', function() {
        updateModelIntro();
        // 모델이 변경되면 세션의 대화 내역을 초기화하는 API 호출
        fetch('/reset_conversation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        }).then(() => {
            // 채팅 창 초기화
            chatMessages.innerHTML = defaultChatLogoHTML;
            // 예시 질문 창 펼치기
            if (questionsContainer.classList.contains('collapsed')) {
                toggleQuestions();
            }
        });
    });

    sendButton.addEventListener('click', sendMessage);

    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

    if (isMobile) {
        chatInput.addEventListener('keydown', function (e) {
            // 모바일에서는 Enter 키가 줄바꿈
            if (e.key === 'Enter' && !e.shiftKey) {
                // 기본 동작 허용 (줄바꿈)
            }
        });
    } else {
        chatInput.addEventListener('keydown', function (e) {
            // 데스크탑에서는 Enter 키로 전송
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    chatInput.addEventListener('input', function () {
        // 입력란의 높이를 자동으로 조절
        chatInput.style.height = 'auto';
        chatInput.style.height = chatInput.scrollHeight + 'px';
    });

    //예시 질문 박스
    const exampleQuestions = document.querySelectorAll('.example-question');

    exampleQuestions.forEach(function (button) {
        button.addEventListener('click', function () {
            const question = this.textContent;
            chatInput.value = question;
            chatInput.style.height = 'auto';
            sendMessage();
        });
    });

    function refreshExampleQuestions() {
        fetch('/get_example_questions')
            .then(response => response.json())
            .then(data => {
                const questionsContainer = document.getElementById('questions-container');
                questionsContainer.innerHTML = '';
                data.example_questions.forEach(question => {
                    const button = document.createElement('button');
                    button.classList.add('example-question');
                    button.textContent = question;
                    button.addEventListener('click', function () {
                        chatInput.value = question;
                        chatInput.style.height = 'auto';
                        sendMessage();
                        refreshExampleQuestions(); // Refresh after sending a message
                    });
                    questionsContainer.appendChild(button);
                });
            });
    }

    // Call refreshExampleQuestions initially
    refreshExampleQuestions();

    const toggleButton = document.getElementById('toggle-questions');
    const questionsContainer = document.getElementById('questions-container');
    toggleButton.classList.add('open');

    function toggleQuestions(){
        if(questionsContainer.style.display === 'none'){
            questionsContainer.style.display = 'flex';
            toggleButton.classList.add('open');
        }else{
            questionsContainer.style.display = 'none';
            toggleButton.classList.remove('open');
        }
    }

    toggleButton.addEventListener('click', toggleQuestions);    

    const menuButton = document.getElementById('menu-button');
    const sideMenu = document.getElementById('side-menu');
    const overlay = document.createElement('div');
    overlay.id = 'overlay';
    document.body.appendChild(overlay);

    menuButton.addEventListener('click', function () {
        sideMenu.classList.add('open');
        overlay.classList.add('show');
        // 오버레이 클릭 시 사이드 메뉴 닫기
        overlay.addEventListener('click', closeSideMenu);
    });
    
    function closeSideMenu() {
        sideMenu.classList.remove('open');
        overlay.classList.remove('show');
        // 이벤트 리스너 제거하여 메모리 누수 방지
        overlay.removeEventListener('click', closeSideMenu);
    }

    function sendMessage() {
        const message = chatInput.value.trim();
        if (message === '') return;

        addMessage('user', message);
        chatInput.value = '';
        chatInput.style.height = 'auto';

        // 타이핑 인디케이터 추가
        showTypingIndicator();

        // 모델 소개 박스 숨기기
        if (modelIntro) {
            modelIntro.style.display = 'none';
        }

        // 모델 선택 드롭다운 비활성화
        modelSelect.disabled = true;
        sendButton.disabled = true;
        sendButton.classList.add('disabled'); // 비활성화 클래스 추가
        //sendButton.querySelector('img').src = '/static/images/send_icon_disabled.png';

        fetch('/chat_api', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                model: modelSelect.value
            })
        })
        .then(response => response.json())
        // 서버 응답 처리 부분 수정
        .then(data => {
            hideTypingIndicator();
            // 모델 선택 드롭다운 및 전송 버튼 다시 활성화
            modelSelect.disabled = false;
            sendButton.disabled = false;
            sendButton.classList.remove('disabled');
        
            if (data.answer) {
                addMessage('bot', data.answer, function() {
                    if (data.reset_message) {
                        addMessage('system', data.reset_message);
                    }
                });
            } else {
                addMessage('bot', '죄송합니다. 응답을 가져올 수 없습니다.', function() {
                    if (data.reset_message) {
                        addMessage('system', data.reset_message);
                    }
                });
            }
        })
        .catch(error => {
            console.error('Error:', error);
            hideTypingIndicator();
            // 모델 선택 드롭다운 다시 활성화
            modelSelect.disabled = false;
            sendButton.disabled = false;
            sendButton.classList.remove('disabled');
            //sendButton.querySelector('img').src = '/static/images/send_icon.png';
            addMessage('bot', '죄송합니다. 오류가 발생했습니다.');
        });
    }

    function addMessage(sender, text, callback) { // 수정된 함수
        if (sender === 'system') {
            const messageElement = document.createElement('div');
            messageElement.classList.add('system-message');
            messageElement.textContent = text;
            chatMessages.appendChild(messageElement);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            return;
        }
        // 로고 숨기기
        const chatLogo = document.getElementById('chat-logo');
        if (chatLogo) {
            chatLogo.style.display = 'none';
        }

        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender);

        const avatar = document.createElement('img');
        avatar.classList.add('avatar');
        if (sender === 'user') {
            avatar.src = '/static/images/user_avatar.png';
        } else {
            avatar.src = botAvatarSrc; // 선택된 모델에 따른 봇 아바타 이미지 사용
        }

        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');

        if (sender === 'bot') {
            typeText(messageContent, text, 0, function() { // 수정된 부분
                if (callback) callback(); // 타이핑 완료 후 콜백 호출
            });
        } else {
            messageContent.textContent = text;
        }

        if (sender === 'user') {
            messageElement.appendChild(messageContent);
            messageElement.appendChild(avatar);
        } else {
            messageElement.appendChild(avatar);
            messageElement.appendChild(messageContent);
        }

        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // 메시지 복사 기능 추가
        messageContent.addEventListener('touchstart', handleTouchStart, {passive: true});
        messageContent.addEventListener('touchend', handleTouchEnd);

        let touchTimer;

        function handleTouchStart(e) {
            touchTimer = setTimeout(() => {
                copyToClipboard(text);
            }, 1000); // 1초 이상 길게 누르면 복사
        }

        function handleTouchEnd(e) {
            if (touchTimer) {
                clearTimeout(touchTimer);
            }
        }
    }

    function copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(() => {
            console.log('복사 성공');
        }).catch(err => {
            console.error('복사 실패', err);
        });
    }

    function typeText(element, text, index = 0, callback) { // 수정된 함수
        if (index < text.length) {
            element.textContent += text.charAt(index);
            setTimeout(() => {
                typeText(element, text, index + 1, callback);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }, 30); // 타이핑 속도 조절 (밀리초)
        } else {
            if (callback) callback(); // 타이핑 완료 시 콜백 호출
        }
    }

    function showTypingIndicator() {
        const typingIndicator = document.createElement('div');
        typingIndicator.classList.add('message', 'bot', 'typing-indicator');

        const avatar = document.createElement('img');
        avatar.classList.add('avatar');
        avatar.src = botAvatarSrc; // 선택된 모델에 따른 봇 아바타 이미지 사용

        const dots = document.createElement('div');
        dots.classList.add('message-content');
        dots.innerHTML = `
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
        `;

        typingIndicator.appendChild(avatar);
        typingIndicator.appendChild(dots);

        typingIndicator.id = 'typing-indicator';
        chatMessages.appendChild(typingIndicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            chatMessages.removeChild(typingIndicator);
        }
    }
});
